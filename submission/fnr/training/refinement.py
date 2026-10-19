"""Offline reproduction of the additional factor statistics and bounded neural head."""
from ..paths import config_path
from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd
from .prepare_extended import prepare_extended
from ..models import peer as model
from ..features.window import panels, NAMES, BAR, FUND
from ..features.window_math import stats, solve_at
from ..features.window_state import state_panel
from .artifacts import write, manifest

def train(data_dir, base_dir, out):
    cfg = json.loads(config_path('new_head_recipe.json').read_text())
    scfg = json.loads(config_path('new_stat_recipe.json').read_text())
    kcfg = json.loads((base_dir / 'statistical_correction/config.json').read_text())
    keep = kcfg['keep']
    _, contract, d, f, neighbors, weights = prepare_extended(data_dir, base_dir / 'factor_initial', cfg)
    columns = {'MarketData': ['日期', '代码', 'open', 'close', 'amount', 'is_suspend'], 'Barra_factor': ['日期', '代码'] + BAR, 'Factor33_winsor': ['日期', '代码', 'turn'] + FUND}
    tables = {k: pd.concat([pd.read_parquet(path, columns=cols) for path in sorted((data_dir / k).glob('*/*/*.parquet')) if '201601' <= path.stem <= '202212'], ignore_index=True) for k, cols in columns.items()}
    p = panels(tables)
    xx = p['x'][:, :, keep]
    raw_stats = stats(xx, 0.5 * np.sign(p['y']), p['label_mask'] & (abs(p['y']) >= 0.45))
    penalty = np.diag([0.7 if i < 10 else 0.1 for i in keep])
    coef = np.zeros((len(xx), len(keep)))
    zero = np.zeros(len(keep))
    for t in range(42, len(xx)):
        coef[t] = solve_at(dict(p, x=xx), t, zero, penalty, raw_stats, dict(lookback=21, window=20, half_life=20))
    context = np.concatenate([xx * coef[:, None, :], np.broadcast_to(state_panel(tables, p)[:, None, :], (*xx.shape[:2], 8)), xx], -1).astype(np.float32)
    nodes = np.zeros((*p['mask'].shape, 42), np.float32)
    con = np.zeros((*d['mask'].shape, context.shape[-1]), np.float32)
    for i, day in enumerate(d['days']):
        t = np.searchsorted(p['days'], day)
        assert p['days'][t] == day
        mask = d['mask'][i]
        ix = np.searchsorted(p['codes'], d['codes'][i, mask])
        assert np.array_equal(p['codes'][ix], d['codes'][i, mask])
        nodes[t, ix] = d['nodes'][i, mask]
        con[i, mask] = context[t, ix]
    for j in range(42):
        z = pd.DataFrame(nodes[:, :, j]).where(p['mask']).rank(axis=1, pct=True)
        nodes[:, :, j] = z.sub(z.mean(axis=1), axis=0).fillna(0).to_numpy(np.float32)
    sx = np.concatenate([p['x'], nodes], -1)[:, :, scfg['keep']]
    target = pd.DataFrame(p['y'] - scfg['residual_gamma'] * nodes[:, :, 32]).where(p['label_mask']).rank(axis=1, pct=True)
    target = target.sub(target.mean(axis=1), axis=0).fillna(0).to_numpy(np.float32)
    a, b = stats(sx, 0.5 * np.sign(target), p['label_mask'] & (abs(target) >= 0.45))
    ix = np.arange(scfg['lookback'] - 1, len(p['days']) - 2)
    ix = ix[p['days'][ix + 2] < 20230101]
    prior = np.linalg.solve(a[ix].mean(0) + scfg['prior_ridge'] * np.eye(sx.shape[-1]), b[ix].mean(0))
    penalty = np.diag([scfg['style_ridge'] if j < 10 else scfg['ridge'] for j in scfg['keep']])
    covariance = np.zeros_like(penalty)
    for t in ix:
        mask = p['mask'][t] & p['mask'][t - 1]
        change = sx[t, mask].astype(float) - sx[t - 1, mask]
        covariance += change.T @ change / max(mask.sum(), 1)
    covariance /= len(ix)
    penalty += scfg['stability'] * covariance / (np.trace(covariance) / len(covariance))
    folder = out / 'statistical'
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / 'prior.npy', prior)
    np.save(folder / 'penalty.npy', penalty)
    write(folder / 'config.json', scfg)
    write(folder / 'training.json', dict(max_label_exit=int(p['days'][ix[-1] + 2]), cutoff=20230101))
    a = model.load_adapter(base_dir / 'factor/numpy_model/adapter')
    b = model.load_adapter(base_dir / 'factor/numpy_model/adapter_r1')
    sa = model.apply_adapter(a, f, d['mask'], d['teacher'])[0]
    sb = model.apply_adapter(b, f, d['mask'], d['teacher'])[0]
    d['teacher'] = (0.475 * sa.astype(float) + 0.525 * sb.astype(float)).astype(np.float32)
    features = np.concatenate([f, con], -1)
    values = features[d['mask']]
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    sd = values.std(0, dtype=np.float64).astype(np.float32)
    sd = np.where(sd > 1e-06, sd, 1)
    pre = dict(mean=mean.tolist(), std=sd.tolist(), clip=8)
    x = np.clip((features - mean) / sd, -8, 8)
    # Preserve the selected-feature array layout used by the frozen training run.
    x = x[:, :, list(range(178))]
    del values, features, nodes, sx, tables, context, con, f
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    cfg['feature_indices'] = list(range(178))
    net = model.build(cfg, x.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    folder = out / 'head'
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'config.json', cfg)
    for epoch in range(cfg['epochs']):
        order = np.random.permutation(len(x))
        losses = []
        for j in range(0, len(order), 16):
            ii = order[j:j + 16]
            batch = {k: torch.from_numpy(np.array(d[k][ii])) for k in ['mask', 'label_mask', 'teacher', 'returns', 'y_rank']}
            batch['neighbors'] = torch.from_numpy(np.array(neighbors[ii], dtype=np.int64))
            batch['weights'] = torch.from_numpy(np.array(weights[ii]))
            opt.zero_grad(set_to_none=True)
            score, delta = net(torch.from_numpy(x[ii]), batch['mask'], batch['teacher'])
            loss, _ = model.loss(score, delta, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)))
        with (folder / 'learning_curve.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    model.export(net, folder / 'adapter', pre)
    write(folder / 'training.json', dict(cutoff=20230101, source_contract=contract, full_public_fit=True))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    train(args.data_dir.resolve(), args.base_dir.resolve(), args.output_dir.resolve())
