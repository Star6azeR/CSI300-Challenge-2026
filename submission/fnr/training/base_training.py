"""Fixed full-history recipe, rebuilt solely from the three public raw tables.

This module never loads an existing model or research cache. Training is explicit;
prepare_training and training_contract do not fit or score any model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import tabm
import torch
from threadpoolctl import threadpool_limits
from ..features.daily import build_features
from .networks import build_tabm
from ..inference.kernels import neural_raw_matrix, neural_matrix, lgbm_matrix, sha256
KEY = ['日期', '代码']
TABLES = ('Factor33_winsor', 'Barra_factor', 'MarketData')
CUTOFF = 20230101
LGBM_PARAMETERS = dict(n_estimators=800, learning_rate=0.035, num_leaves=15, max_depth=-1, min_child_samples=500, reg_lambda=20, colsample_bytree=0.85, subsample=0.8, subsample_freq=1, random_state=42, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
TABM_PARAMETERS = dict(epochs=24, batch_size=4096, learning_rate=0.002, weight_decay=0.0003, dropout=0.1, seed=42, n_jobs=4, predict_batch_size=4096, inference_kernel_batch_size=256, raw_score_round_decimals=6, max_grad_norm=5.0, predict_by_date=False, d_block=64, n_blocks=2, k=8, arch_type='tabm', num_embeddings='none', embedding_knots=8, d_embedding=4)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_public_tables(data_dir):
    """Read exactly 2016--2022 monthly files beneath the explicit data directory."""
    root = Path(data_dir).resolve()
    tables = {}
    for name in TABLES:
        pieces = []
        for year in range(2016, 2023):
            for month in range(1, 13):
                path = root / name / str(year) / f'{month:02d}' / f'{year}{month:02d}.parquet'
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Raw table path leaves the supplied data directory')
                frame = pd.read_parquet(path)
                if not set(KEY).issubset(frame) or not frame.columns.is_unique:
                    raise ValueError('Missing/duplicate raw columns: ' + name)
                if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
                    raise ValueError('Missing/duplicate raw row keys: ' + name)
                if not pd.api.types.is_integer_dtype(frame['日期']):
                    raise ValueError('Raw dates must be integer YYYYMMDD')
                if not (frame['日期'] // 100).eq(year * 100 + month).all():
                    raise ValueError('Raw monthly file contains a date outside its declared month')
                pieces.append(frame)
        tables[name] = pd.concat(pieces, ignore_index=True).sort_values(KEY).reset_index(drop=True)
        if tables[name].duplicated(KEY).any():
            raise ValueError('Duplicated raw table keys: ' + name)
    return tables

def h1_labels(market, barra):
    """Next exchange-day open to second-next open; both endpoints tradable."""
    dates = sorted(market['日期'].unique())
    frame = barra[KEY].copy()
    frame['entry_date'] = frame['日期'].map(dict(zip(dates[:-1], dates[1:])))
    frame['exit_date'] = frame['日期'].map(dict(zip(dates[:-2], dates[2:])))
    for endpoint in ('entry', 'exit'):
        source = market[KEY + ['open', 'is_suspend']].rename(columns={'日期': endpoint + '_date', 'open': endpoint + '_open', 'is_suspend': endpoint + '_suspend'})
        frame = frame.merge(source, on=[endpoint + '_date', '代码'], how='left', validate='many_to_one')
    good = frame.entry_suspend.eq(0) & frame.exit_suspend.eq(0) & frame.entry_open.gt(0) & frame.exit_open.gt(0) & np.isfinite(frame.entry_open) & np.isfinite(frame.exit_open)
    frame['future_return'] = (frame.exit_open / frame.entry_open - 1).where(good)
    frame['target'] = frame.groupby('日期').future_return.rank(pct=True)
    return frame[KEY + ['entry_date', 'exit_date', 'future_return', 'target']]

def prepare_training(data_dir):
    tables = read_public_tables(data_dir)
    with threadpool_limits(limits=4):
        full, groups = build_features(tables['Factor33_winsor'], tables['Barra_factor'], tables['MarketData'])
    columns = list(groups['mixed61'])
    if len(columns) != 347 or len(set(columns)) != 347:
        raise ValueError('Frozen rich mixed61 recipe requires exactly 347 ordered columns')
    labels = h1_labels(tables['MarketData'], tables['Barra_factor'])
    full = full[KEY + columns].merge(labels, on=KEY, how='left', validate='one_to_one')
    selected = full['日期'].ge(20160101) & full.exit_date.lt(CUTOFF) & np.isfinite(full.future_return)
    train = full.loc[selected].sort_values(KEY, kind='stable').reset_index(drop=True)
    if train.empty or train.duplicated(KEY).any() or train.exit_date.le(train['日期']).any():
        raise ValueError('Empty, duplicate or invalid training rows')
    if any((str(train[c].dtype) != 'float32' for c in columns)):
        raise ValueError('Frozen rich feature precision must be float32')
    return (train, columns)

def fit_preprocessing(train, columns):
    """Median/mean/population-SD learned only after the strict label cutoff."""
    values = neural_raw_matrix(train, columns, {c: str(train[c].dtype) for c in columns})
    imputation = pd.DataFrame(values, columns=columns).median().fillna(0).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, imputation)
    mean, std = (values.mean(axis=0), values.std(axis=0, ddof=0))
    return dict(fillna=dict(zip(columns, imputation.tolist())), fill_strategy='training_median; all-missing=0', standardize=True, mean=mean.tolist(), scale=np.where(std > 1e-12, std, 1.0).tolist(), clip_bounds=None, statistics_fit_only_after_label_cutoff=True, matrix_dtype='float32', preprocessing_dtype='float64')

def targets_and_weights(train):
    target = train['target'].to_numpy(dtype='float64') - 0.5
    weights = 1 / train.groupby('日期')['代码'].transform('size')
    weights /= weights.mean()
    if not np.isfinite(target).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite target or sample weight')
    return (target, weights)

def training_contract(train, columns, preprocessing):
    """Reproducible value fingerprints; no fit, prediction, or metric evaluation."""
    target, weight = targets_and_weights(train)
    neural_weight = 1.0 / train.groupby('日期')['代码'].transform('size').to_numpy(dtype=float)
    neural_weight /= neural_weight.mean()
    matrix_digest = hashlib.sha256()
    for column in columns:
        values = train[column].to_numpy(dtype='<f4', copy=True)
        values[np.isnan(values)] = np.nan
        matrix_digest.update(column.encode('utf-8') + b'\x00' + values.tobytes())
    key_digest = hashlib.sha256(pd.util.hash_pandas_object(train[KEY], index=False).to_numpy().tobytes()).hexdigest()
    return dict(train_rows=len(train), feature_count=len(columns), train_signal_start=int(train['日期'].min()), train_signal_end=int(train['日期'].max()), train_label_exit_max=int(train.exit_date.max()), train_label_cutoff_exclusive=CUTOFF, row_keys_sha256=key_digest, ordered_feature_values_sha256=matrix_digest.hexdigest(), target_float64_sha256=hashlib.sha256(target.astype('<f8').tobytes()).hexdigest(), weights_float64_sha256=hashlib.sha256(weight.to_numpy(dtype='<f8').tobytes()).hexdigest(), tabm_target_float32_sha256=hashlib.sha256(target.astype('<f4').tobytes()).hexdigest(), tabm_weights_float32_sha256=hashlib.sha256(neural_weight.astype('<f4').tobytes()).hexdigest(), preprocessing_sha256=hashlib.sha256(json.dumps(preprocessing, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest(), fingerprint_note='Feature hash: ordered UTF-8 column + NUL + float32 little-endian column bytes, canonical NaN; keys use pandas hash_pandas_object', fit_performed=False, metrics_or_backtest_performed=False)

def component_metadata(train, columns, family, preprocessing):
    parameters = deepcopy(LGBM_PARAMETERS if family == 'lgbm' else TABM_PARAMETERS)
    config = dict(name='full_history_' + family, model=family, source='rich', new_features=False, feature_groups=['mixed61'], columns=columns, target='rank', rank_center=0.5, train_start=20160101, precision='preserve', smooth=1, rank_scores=True, score_round_decimals=12, params=deepcopy(parameters), year=2023)
    meta = dict(format_version=1, family=family, features=columns, config=config, feature_precision='preserve', feature_dtypes={c: str(train[c].dtype) for c in columns}, preprocessing=preprocessing, parameters=parameters, prediction_dtype='float64', evaluation_year=2023, train_label_cutoff_exclusive=CUTOFF, train_rows=len(train), train_signal_start=int(train['日期'].min()), train_signal_end=int(train['日期'].max()), train_label_exit_max=int(train.exit_date.max()), label='O(t+2)/O(t+1)-1', horizon=1, target_transform=dict(kind='rank', rank_center=0.5, parameters={}), sample_weight='inverse training-day stock count, normalized mean 1', artifact_role='deployment_refit', evaluation_performed=False, training_source_sha256=sha256(Path(__file__)), versions=dict(numpy=np.__version__, pandas=pd.__version__, lightgbm=lgb.__version__, torch=torch.__version__, tabm=tabm.__version__))
    if family == 'torch_tabm':
        meta.update(deep_format_version=1, architecture=dict(family=family, input_dim=len(columns), dropout=0.1, d_block=64, n_blocks=2, k=8, arch_type='tabm', activation='ReLU', start_scaling_init='random-signs', num_embeddings=None), prediction_dtype='float64 (quantized from float32 inference)', validation_split=None, early_stopping=False, device='cpu', n_jobs=4, ensemble_size=8, training_loss='mean_over_rows(weight * mean_over_heads(squared_error))', inference_aggregation='mean_regression_heads_before_frozen_raw_rounding')
    return meta

def train_components(train, columns, preprocessing, destination, contract):
    """Actually train the fixed two components; only called without --validate-only."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Model output must be absent or empty; existing frozen artifacts are never overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('lgbm', 'tabm'):
        (destination / name).mkdir()
    y, weights = targets_and_weights(train)
    tree_meta = component_metadata(train, columns, 'lgbm', dict(fillna=None, standardize=False, clip_bounds=None))
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        tree = lgb.LGBMRegressor(**LGBM_PARAMETERS)
        tree.fit(lgbm_matrix(train, tree_meta), y, sample_weight=weights)
    tree_meta['fit_seconds'] = time.perf_counter() - started
    artifact = destination / 'lgbm' / 'model.txt'
    tree.booster_.save_model(str(artifact))
    tree_meta.update(artifact='model.txt', artifact_sha256=sha256(artifact))
    write_json(destination / 'lgbm' / 'model_meta.json', tree_meta)
    del tree
    meta = component_metadata(train, columns, 'torch_tabm', preprocessing)
    x = torch.from_numpy(neural_matrix(train, meta))
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
    # Match the original neural path's NumPy reduction, separately from pandas tree weights.
    neural_weights = 1.0 / train.groupby('日期')['代码'].transform('size').to_numpy(dtype=float)
    neural_weights /= neural_weights.mean()
    w = torch.from_numpy(neural_weights.astype(np.float32))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model = build_tabm(meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003, foreach=False)
    rng = np.random.default_rng(42)
    history = []
    with threadpool_limits(limits=4):
        for epoch in range(24):
            epoch_started = time.perf_counter()
            model.train(True)
            permutation = rng.permutation(len(train))
            accumulated = 0.0
            for begin in range(0, len(train), 4096):
                index = torch.from_numpy(permutation[begin:begin + 4096])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[index])
                if prediction.shape != (len(index), 8, 1):
                    raise ValueError('TabM must produce (batch,8,1)')
                member_errors = (prediction.squeeze(-1) - y_tensor[index, None]).square()
                loss = (w[index] * member_errors.mean(dim=1)).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite TabM training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                accumulated += float(loss.detach()) * len(index)
            history.append(dict(epoch=epoch + 1, weighted_online_training_mse=accumulated / len(train), seconds=time.perf_counter() - epoch_started))
    model.train(False)
    if any((not torch.isfinite(parameter).all().item() for parameter in model.parameters())):
        raise FloatingPointError('Nonfinite trained TabM parameters')
    meta.update(train_history=history, fit_seconds=time.perf_counter() - started, parameter_count=sum((p.numel() for p in model.parameters())), determinism=dict(seed=42, cpu_threads=4, deterministic_algorithms=True, training_shuffle='seeded training rows only', inference_fixed_kernel_batch=256, canonical_prediction_by_date=False, raw_score_round_decimals=6, cross_platform_bitwise_guarantee=False))
    artifact = destination / 'tabm' / 'model.npz'
    arrays = {name: tensor.detach().cpu().contiguous().numpy().copy() for name, tensor in model.state_dict().items()}
    np.savez(artifact, **arrays)
    meta.update(artifact='model.npz', artifact_sha256=sha256(artifact), artifact_format='numpy_npz_tensor_state_dict')
    write_json(destination / 'tabm' / 'model_meta.json', meta)
    ensemble = dict(format_version=1, family='lgbm_tabm_rank_blend', component_order=['lgbm', 'tabm'], weights=[0.5, 0.5], rank_components=True, score_round_decimals=12, lookback=61, features=columns, train_label_exit_max=int(train.exit_date.max()), train_label_cutoff_exclusive=CUTOFF, application_year=2023, evaluation_performed=False, artifact_role='deployment_refit')
    write_json(destination / 'ensemble.json', ensemble)
    contract = dict(contract, fit_performed=True)
    write_json(destination / 'training_input_contract.json', contract)
    files = ['lgbm/model.txt', 'lgbm/model_meta.json', 'tabm/model.npz', 'tabm/model_meta.json', 'ensemble.json', 'training_input_contract.json']
    write_json(destination / 'manifest.json', dict(format_version=1, files_sha256={name: sha256(destination / name) for name in files}))
    return dict(model_dir=str(destination), train_rows=len(train), components=['lgbm', 'tabm'], fit_performed=True)
