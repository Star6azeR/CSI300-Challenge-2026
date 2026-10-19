"""Train a frozen-factor peer adapter from the three public raw tables.

Run after the initial FA004 training stage. No research cache or path is used.
The original factor directory is retained as factor_initial on success.
"""
from ..paths import config_path
from pathlib import Path
import argparse
import hashlib
import json
import random
import resource
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
from ..models import peer as model
from .data import DailyData, KEY
from ..features.daily import build_features
from .artifacts import manifest, sha, write
from .base_training import h1_labels, read_public_tables

def array_hash(value):
    """Logical array fingerprint, independent of NPY/NPZ container metadata."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()

def adapter_preprocessing(features, mask):
    # Keep the same reduction order and intermediate dtypes as study.py.
    values = features[mask]
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    std = values.std(0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-06, std, 1).astype(np.float32)
    return dict(mean=mean.tolist(), std=std.tolist(), clip=8)

def prepare_raw(data_dir, source, cfg):
    pre = json.loads((source / 'preprocessing.json').read_text())
    columns = pre['features']
    if len(columns) != 347 or len(set(columns)) != 347:
        raise ValueError('The frozen teacher requires 347 ordered features')
    tables = read_public_tables(data_dir)
    full, _ = build_features(tables['Factor33_winsor'], tables['Barra_factor'], tables['MarketData'])
    labels = h1_labels(tables['MarketData'], tables['Barra_factor'])
    selected = full['日期'].ge(20160101) & full['日期'].lt(20230101)
    frame = full.loc[selected, KEY + columns].sort_values(KEY).reset_index(drop=True)
    # Match the research treatment: selected raw feature is zeroed before the
    # original frozen mean/std transform, not after standardization.
    frame['is_st'] = 0.0
    legal = labels.loc[labels['日期'].ge(20160101) & labels['日期'].lt(20230101) & labels.exit_date.lt(20230101), KEY + ['entry_date', 'exit_date', 'future_return']]
    if legal.empty or not legal.exit_date.lt(20230101).all():
        raise ValueError('No legal pre-2023 H1 labels')
    if not (legal.entry_date.gt(legal['日期']).all() and legal.exit_date.gt(legal.entry_date).all()):
        raise ValueError('Invalid H1 label endpoints')
    frame = frame.merge(legal, on=KEY, how='left', validate='one_to_one')
    if frame.empty or frame.duplicated(KEY).any():
        raise ValueError('Empty or duplicated asof feature rows')
    contract = dict(split='train', year=2023, start=20160101, cutoff_exclusive=20230101, label_exit_max=int(legal.exit_date.max()), asof_rows=len(frame), valid_labels=int(np.isfinite(frame.future_return).sum()), is_st='raw zero', H=1, label='adjusted O(t+2)/O(t+1)-1', input_lookback=61, membership='asof only, independent of labels', teacher_sha256={str(p.relative_to(source / 'numpy_model')): sha(p) for p in sorted((source / 'numpy_model').rglob('*')) if p.is_file()}, preprocessing_sha256=sha(source / 'preprocessing.json'))
    data = DailyData(frame, pre, history_length=0)
    del tables, full, labels, legal, frame
    teacher = model.core.load_numpy(source / 'numpy_model')
    mode = cfg['neighborhood']
    graph_key = 'barra' if mode == 'none' else mode
    n, width = data.mask.shape
    nodes = np.zeros((n, width, 42), np.float32)
    scores = np.zeros((n, width), np.float32)
    indices = np.zeros((n, width, 24), np.int16)
    weights = np.zeros(indices.shape, np.float32)
    for i in range(n):
        nodes[i], scores[i], graphs = model.encode(teacher, data.x[i:i + 1], data.mask[i:i + 1])
        indices[i], weights[i] = graphs[graph_key]
    packed = dict(mask=data.mask, label_mask=data.label_mask, returns=data.raw_return, y_rank=data.y_rank, teacher=scores, days=data.days)
    contract['prepared_array_sha256'] = {key: array_hash(value) for key, value in packed.items()}
    contract['prepared_array_sha256'].update(nodes=array_hash(nodes), neighbors=array_hash(indices), weights=array_hash(weights))
    del data, teacher
    features = model.features_many(nodes, indices, weights, mode)
    del nodes
    return (pre, contract, packed, features, indices, weights)

def train(root, data_dir, cfg, pending):
    source = root / 'factor'
    started = time.perf_counter()
    pre, contract, data, features, indices, weights = prepare_raw(data_dir, source, cfg)
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # The study seeds only after deterministic frozen-teacher preparation.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    feature_pre = adapter_preprocessing(features, data['mask'])
    x = np.clip((features - np.asarray(feature_pre['mean'], np.float32)) / np.asarray(feature_pre['std'], np.float32), -8, 8)
    if not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite prepared adapter inputs')
    net = model.build(cfg, x.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    write(pending / 'config.json', cfg)
    write(pending / 'feature_preprocessing.json', feature_pre)
    write(pending / 'training_contract.json', contract)
    initial = net(torch.from_numpy(x[:2]), torch.from_numpy(np.array(data['mask'][:2])), torch.from_numpy(np.array(data['teacher'][:2])))[0].detach().numpy()
    if not np.array_equal(initial, data['teacher'][:2]):
        raise AssertionError('The new adapter must start at the frozen teacher')
    records = []
    for epoch in range(cfg['epochs']):
        values = []
        order = np.random.permutation(len(x))
        for j in range(0, len(order), cfg['batch_days']):
            ii = order[j:j + cfg['batch_days']]
            batch = {key: torch.from_numpy(np.array(data[key][ii])) for key in ('mask', 'label_mask', 'teacher', 'returns', 'y_rank')}
            batch['neighbors'] = torch.from_numpy(np.array(indices[ii], dtype=np.int64))
            batch['weights'] = torch.from_numpy(np.array(weights[ii]))
            opt.zero_grad(set_to_none=True)
            scores, delta = net(torch.from_numpy(x[ii]), batch['mask'], batch['teacher'])
            loss, parts = model.loss(scores, delta, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite adapter loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            values.append([float(loss.detach()), *[float(v.detach()) for v in parts.values()], float(delta.detach().abs().mean())])
        row = dict(epoch=epoch + 1, loss_terms=['total', 'rank', 'preserve', 'pair', 'mean_abs_delta'], values=np.mean(values, 0).tolist())
        records.append(row)
        with (pending / 'learning_curve.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    net.eval()
    numpy_model = pending / 'numpy_model'
    model.export(net, numpy_model / 'adapter', feature_pre)
    loaded = model.load_adapter(numpy_model / 'adapter')
    index = len(x) // 2
    with torch.inference_mode():
        reference = net(torch.from_numpy(x[index:index + 1]), torch.from_numpy(np.array(data['mask'][index:index + 1])), torch.from_numpy(np.array(data['teacher'][index:index + 1])))[0].numpy()
    predicted = model.apply_adapter(loaded, features[index:index + 1], data['mask'][index:index + 1], data['teacher'][index:index + 1])[0]
    parity = float(np.max(abs(reference - predicted)))
    if parity >= 1e-06:
        raise AssertionError('NumPy export parity failed: ' + str(parity))
    shutil.copytree(source / 'numpy_model', numpy_model / 'teacher')
    # Keep the original preprocessing bytes as well as their numerical values.
    shutil.copyfile(source / 'preprocessing.json', pending / 'preprocessing.json')
    completion = dict(candidate=cfg.get('name'), year=2023, config=cfg, contract=contract, training_contract=contract, new_parameters=sum((p.numel() for p in net.parameters())), teacher_parameters_frozen=True, initial_prediction_parity=0.0, numpy_torch_max_error=parity, seconds=time.perf_counter() - started, training_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024), adapter_sha256={p.name: sha(p) for p in (numpy_model / 'adapter').iterdir()}, versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__), cross_platform_bitwise_guarantee=False)
    write(pending / 'training_completion.json', completion)
    return completion

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.model_dir.resolve()
    source, initial = (root / 'factor', root / 'factor_initial')
    pending = root / 'factor_peer_pending'
    if not (source / 'numpy_model').is_dir() or not (source / 'preprocessing.json').is_file():
        raise ValueError('Expected the trained initial FA004 model at model-dir/factor')
    if initial.exists() or pending.exists():
        raise FileExistsError('Preserve existing factor_initial/factor_peer_pending artifacts')
    cfg = json.loads(config_path('peer_recipe.json').read_text())
    required = set(model.config('P01'))
    if not required.issubset(cfg):
        raise ValueError('peer_recipe.json must contain the full adapter configuration')
    if cfg['epochs'] != 16 or cfg['seed'] != 42 or cfg['neighbors'] != 24:
        raise ValueError('Frozen peer recipe requires 16 epochs, seed 42 and 24 neighbors')
    if cfg['neighborhood'] not in ('none', 'barra', 'factor_barra'):
        raise ValueError('Unknown peer neighborhood')
    pending.mkdir()
    try:
        completion = train(root, args.data_dir, cfg, pending)
        source.rename(initial)
        try:
            pending.rename(source)
        except BaseException:
            initial.rename(source)
            raise
        manifest(root)
    except BaseException:
        # Existing teacher weights survive failed fitting. Partial traces are
        # retained, and a rerun will not silently overwrite them.
        failure_dir = pending if pending.exists() else source
        (failure_dir / 'peer_training_failure.txt').write_text(traceback.format_exc())
        raise
    print(json.dumps(dict(training_completed=True, stage='frozen_factor_peer_adapter', new_parameters=completion['new_parameters'], numpy_torch_max_error=completion['numpy_torch_max_error'])), flush=True)
if __name__ == '__main__':
    main()
