"""Offline training of the frozen factor prior and temporal-stability penalty."""
from ..paths import config_path
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from ..models.statistical import panels, NAMES, BAR, FUND

def train(data_dir, output_dir, config_path, cutoff=20230101):
    cfg = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    cols = {'MarketData': ['日期', '代码', 'open', 'close', 'amount', 'is_suspend'], 'Barra_factor': ['日期', '代码'] + BAR, 'Factor33_winsor': ['日期', '代码', 'turn'] + FUND}
    data = {k: pd.concat([pd.read_parquet(f, columns=v) for f in sorted((data_dir / k).glob('*/*/*.parquet')) if '201601' <= f.stem <= '202212'], ignore_index=True) for k, v in cols.items()}
    p = panels(data)
    x = p['x'][:, :, cfg['keep']]
    dim = x.shape[-1]
    mask = p['label_mask'] & (abs(p['y']) >= 0.45)
    y = 0.5 * np.sign(p['y'])
    ix = np.arange(cfg['lookback'] - 1, len(p['days']) - 2)
    ix = ix[p['days'][ix + 2] < cutoff]
    a = []
    b = []
    for t in ix:
        xx = x[t, mask[t]].astype(np.float64)
        yy = y[t, mask[t]]
        n = max(len(xx), 1)
        a.append(xx.T @ xx / n)
        b.append(xx.T @ yy / n)
    prior = np.linalg.solve(np.array(a).mean(0) + cfg['prior_ridge'] * np.eye(dim), np.array(b).mean(0))
    names = [NAMES[i] for i in cfg['keep']]
    penalty = cfg['ridge'] * np.eye(dim)
    for j, name in enumerate(names):
        if name.startswith('barra_'):
            penalty[j, j] = cfg['style_ridge']
    cov = np.zeros((dim, dim))
    for t in ix:
        m = p['mask'][t] & p['mask'][t - 1]
        dx = x[t, m].astype(float) - x[t - 1, m]
        cov += dx.T @ dx / max(m.sum(), 1)
    cov /= len(ix)
    penalty += cfg['stability'] * cov / max(np.trace(cov) / dim, 1e-12)
    np.save(output_dir / 'prior.npy', prior)
    np.save(output_dir / 'penalty.npy', penalty)
    (output_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
    (output_dir / 'training.json').write_text(json.dumps(dict(features=names, train_max_label_exit=int(p['days'][ix[-1] + 2]), data_end=int(p['days'][-1]), training_cutoff=cutoff), ensure_ascii=False, indent=2))
    print('model saved', output_dir)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=config_path('statistical_recipe.json'))
    ap.add_argument('--cutoff', type=int, default=20230101)
    args = ap.parse_args()
    train(args.data_dir, args.output_dir, args.config, args.cutoff)
