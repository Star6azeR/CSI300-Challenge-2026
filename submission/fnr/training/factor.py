"""Fit the masked full-asof daily auxiliary model; export pure NumPy weights."""
from ..paths import config_path
from pathlib import Path
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from .base_training import read_public_tables, h1_labels
from ..features.daily import build_features
from .data import DailyData, fit_preprocessing, KEY
from ..models import factor_model
from .artifacts import manifest, write

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(config_path('auxiliary_recipe.json').read_text())
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    output = args.model_dir / 'factor'
    if output.exists():
        raise FileExistsError('Refusing to overwrite auxiliary output')
    if not (args.model_dir / 'ensemble.json').exists():
        raise ValueError('Train the base ensemble first')
    tables = read_public_tables(args.data_dir)
    full, groups = build_features(tables['Factor33_winsor'], tables['Barra_factor'], tables['MarketData'])
    columns = list(groups['mixed61'])
    expected = json.loads((args.model_dir / 'ensemble.json').read_text())['features']
    if columns != expected:
        raise ValueError('Raw feature order differs from frozen recipe')
    labels = h1_labels(tables['MarketData'], tables['Barra_factor'])
    # Membership is defined before labels are joined. Invalid/future labels do
    # not remove a stock from the attention pool or the normalization sample.
    frame = full.loc[full['日期'].ge(20160101) & full['日期'].lt(20230101), KEY + columns + ['raw_float_market_cap']].sort_values(KEY).reset_index(drop=True)
    legal = labels.loc[labels.exit_date.lt(20230101), KEY + ['entry_date', 'exit_date', 'future_return']]
    aligned = frame[KEY].merge(legal, on=KEY, how='left', validate='one_to_one')
    for name in ('entry_date', 'exit_date', 'future_return'):
        frame[name] = aligned[name].to_numpy()
    if len(frame) == 0 or frame.duplicated(KEY).any():
        raise ValueError('Empty or duplicate asof pool')
    finite = np.isfinite(frame.future_return)
    if not finite.any() or not frame.loc[finite, 'exit_date'].lt(20230101).all():
        raise ValueError('Invalid label cutoff')
    pre = fit_preprocessing(frame, columns)
    contract = {'H': 1, 'label': 'adjusted O(t+2)/O(t+1)-1', 'train_start': 20160101, 'train_signal_end': int(frame['日期'].max()), 'train_label_exit_max': int(legal.exit_date.max()), 'label_cutoff_exclusive': 20230101, 'asof_feature_rows': len(frame), 'valid_training_labels': int(finite.sum()), 'label_mask_controls_membership': False, 'feature_lookback': 61, 'feature_count': len(columns), 'window_years': None}
    data = DailyData(frame, pre, history_length=0)
    del full, frame, tables, aligned, labels, legal
    model = factor_model.build(cfg, data.dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    output.mkdir(parents=True)
    with (output / 'learning_curve.jsonl').open('w') as log:
        for epoch in range(cfg['epochs']):
            model.train()
            order = np.random.permutation(len(data.groups))
            losses = []
            diagnostics = {}
            started = time.perf_counter()
            for start in range(0, len(order), cfg['batch_days']):
                batch = data.batch(order[start:start + cfg['batch_days']], torch)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch['x'], batch['mask'], history=None)
                loss = factor_model.training_loss(prediction, batch['y_rank'], batch['y_cash'], batch['label_mask'], cfg, context={'raw_return': batch['raw_return']})
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                for name, value in prediction.get('loss_diagnostics', {}).items():
                    diagnostics.setdefault(name, []).append(float(value))
            row = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'seconds': time.perf_counter() - started, 'loss_diagnostics': {name: float(np.mean(values)) for name, values in diagnostics.items()}}
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
    model.eval()
    factor_model.export_numpy(model, output / 'numpy_model', cfg)
    write(output / 'preprocessing.json', pre)
    write(output / 'config.json', cfg)
    write(output / 'training_contract.json', contract)
    write(args.model_dir / 'gaussian_policy.json', {'auxiliary_weight': 0.4, 'lookback_days': 62, 'score_policy': 'rank_blend'})
    manifest(args.model_dir)
    print(json.dumps({'training_completed': True, 'components': ['lgbm', 'tabm', 'ple_rank', 'ple_residual', 'factor']}))
if __name__ == '__main__':
    main()
