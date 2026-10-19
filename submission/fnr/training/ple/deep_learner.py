"""CPU tabular neural learners with train-only preprocessing and annual cutoffs.

Same fit/load/save/predict contract as learner.py. No data loading, budget writes,
validation split, early stopping, or historical experiment runs occur here.
Torch is imported lazily from the installed offline training dependency.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .learner import resolve_features, target_values
KEY = ['日期', '代码']
FAMILIES = {'torch_mlp', 'torch_resnet', 'torch_tabm', 'torch_dcn'}
DEFAULTS = dict(epochs=8, batch_size=4096, learning_rate=0.001, weight_decay=0.0001, dropout=0.1, seed=42, n_jobs=2, predict_batch_size=4096, inference_kernel_batch_size=256, raw_score_round_decimals=6, max_grad_norm=5.0, predict_by_date=False)

def _torch():
    """Use the installed offline training dependency."""
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch

def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def _tabm():
    import tabm
    return tabm

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)

def _parameters(config, width):
    family = config.get('model')
    if family not in FAMILIES:
        raise ValueError('model must be torch_mlp, torch_resnet, torch_tabm or torch_dcn')
    params = deepcopy(DEFAULTS)
    architecture_defaults = {'hidden_dims': [128, 64], 'layer_norm': True} if family == 'torch_mlp' else {'hidden_dim': 128, 'n_blocks': 2, 'expansion_factor': 2} if family == 'torch_resnet' else {'d_block': 64, 'n_blocks': 2, 'k': 8, 'arch_type': 'tabm', 'num_embeddings': 'none', 'embedding_knots': 8, 'd_embedding': 4} if family == 'torch_tabm' else {'n_cross_layers': 2, 'cross_rank': 8, 'dropout': 0.0}
    params.update(architecture_defaults)
    user = deepcopy(config.get('params', {}))
    unknown = set(user) - set(params)
    if unknown:
        raise ValueError(f'Unknown {family} parameters: {sorted(unknown)}')
    params.update(user)
    for name in ('epochs', 'batch_size', 'predict_batch_size', 'inference_kernel_batch_size'):
        params[name] = _positive_int(params[name], name)
    if params['n_jobs'] != 2:
        raise ValueError('PLE deployment runs use exactly two CPU intra-op threads')
    if type(params['predict_by_date']) is not bool:
        raise ValueError('predict_by_date must be boolean')
    if isinstance(params['seed'], bool) or not isinstance(params['seed'], int) or (not 0 <= params['seed'] < 2 ** 32):
        raise ValueError('seed must be an integer in [0,2**32)')
    for name in ('learning_rate', 'max_grad_norm'):
        params[name] = float(params[name])
        if not math.isfinite(params[name]) or params[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    params['weight_decay'], params['dropout'] = (float(params['weight_decay']), float(params['dropout']))
    if not math.isfinite(params['weight_decay']) or params['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if not 0 <= params['dropout'] < 1:
        raise ValueError('dropout must be in [0,1)')
    rounding = params['raw_score_round_decimals']
    if isinstance(rounding, bool) or not isinstance(rounding, int) or (not 0 <= rounding <= 12):
        raise ValueError('raw_score_round_decimals must be an integer from 0 to 12')
    architecture = dict(family=family, input_dim=int(width), dropout=params['dropout'])
    if family == 'torch_mlp':
        dims = params['hidden_dims']
        if not isinstance(dims, (list, tuple)) or not dims:
            raise ValueError('hidden_dims must be a nonempty list')
        params['hidden_dims'] = [_positive_int(d, 'hidden dimension') for d in dims]
        if type(params['layer_norm']) is not bool:
            raise ValueError('layer_norm must be boolean')
        architecture.update(hidden_dims=params['hidden_dims'], layer_norm=params['layer_norm'])
    elif family == 'torch_resnet':
        for name in ('hidden_dim', 'n_blocks', 'expansion_factor'):
            params[name] = _positive_int(params[name], name)
        architecture.update({name: params[name] for name in ('hidden_dim', 'n_blocks', 'expansion_factor')})
    elif family == 'torch_tabm':
        for name in ('d_block', 'n_blocks', 'k'):
            params[name] = _positive_int(params[name], name)
        if params['arch_type'] != 'tabm':
            raise ValueError("This fixed TabM implementation uses official arch_type='tabm'")
        architecture.update({name: params[name] for name in ('d_block', 'n_blocks', 'k', 'arch_type')})
        architecture.update(activation='ReLU', start_scaling_init='random-signs', num_embeddings=None)
        if params['num_embeddings'] not in {'none', 'piecewise_linear'}:
            raise ValueError("num_embeddings must be 'none' or 'piecewise_linear'")
        for name in ('embedding_knots', 'd_embedding'):
            params[name] = _positive_int(params[name], name)
        if params['embedding_knots'] < 2:
            raise ValueError('At least two quantile knots required')
        if params['num_embeddings'] == 'piecewise_linear':
            architecture['num_embeddings'] = dict(kind='piecewise_linear', version='B', activation=False, n_quantile_knots=params['embedding_knots'], d_embedding=params['d_embedding'])
    else:
        for name in ('n_cross_layers', 'cross_rank'):
            params[name] = _positive_int(params[name], name)
        if params['cross_rank'] > width:
            raise ValueError('cross_rank must not exceed input feature width')
        if params['dropout'] != 0.0:
            raise ValueError('The initial cross-only DCN has dropout=0; no hidden deep branch')
        architecture.update(n_cross_layers=params['n_cross_layers'], cross_rank=params['cross_rank'], head='linear', variant='lowrank_dcnv2_style_cross_only')
    return (params, architecture)

def _fit_embedding_bins(values, parameters):
    """Fit boundaries solely on the already filtered/preprocessed training matrix."""
    bins, constant_features = ([], [])
    quantiles = np.linspace(0.0, 1.0, parameters['embedding_knots'])
    for index, column in enumerate(values.T):
        edges = np.unique(np.quantile(column.astype(np.float64), quantiles).astype(np.float32))
        if len(edges) == 1:
            constant_features.append(index)
            center = float(edges[0])
            delta = max(1.0, abs(center) * 0.01)
            bound = np.finfo(np.float32).max
            edges = np.unique(np.array([max(-bound, center - delta), min(bound, center + delta)], dtype=np.float32))
        if len(edges) < 2 or not np.isfinite(edges).all() or (not (np.diff(edges.astype(float)) > 0).all()):
            raise ValueError('Could not construct finite increasing training-only embedding edges')
        bins.append(edges.tolist())
    return dict(bins=bins, constant_feature_indices=constant_features, quantile_positions=quantiles.tolist(), method='numpy.quantile_linear_on_training_float32_matrix; unique_float32', constant_policy='center +/- max(1,abs(center)*0.01), clipped to finite float32', fitted_rows=len(values), statistics_fit_only_after_label_cutoff=True, extrapolation='official piecewise-linear edge extrapolation; no additional clipping')

def _network(torch, architecture, embedding_metadata=None):
    """No BatchNorm: inference of one row does not use other rows' statistics."""
    nn = torch.nn
    if architecture['family'] == 'torch_tabm':
        official = _tabm()
        embedding = None
        spec = architecture['num_embeddings']
        if spec is not None:
            if embedding_metadata is None or len(embedding_metadata['bins']) != architecture['input_dim']:
                raise ValueError('Frozen TabM numerical embedding bins required for every feature')
            bins = [torch.tensor(edges, dtype=torch.float32) for edges in embedding_metadata['bins']]
            import rtdl_num_embeddings
            embedding = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, spec['d_embedding'], activation=spec['activation'], version=spec['version'])
        return official.TabM.make(n_num_features=architecture['input_dim'], cat_cardinalities=None, d_out=1, num_embeddings=embedding, n_blocks=architecture['n_blocks'], d_block=architecture['d_block'], dropout=architecture['dropout'], k=architecture['k'], arch_type=architecture['arch_type'], activation=architecture['activation'], start_scaling_init=architecture['start_scaling_init'])
    if architecture['family'] == 'torch_dcn':

        class LowRankCrossOnly(nn.Module):

            def __init__(self):
                super().__init__()
                width, rank = (architecture['input_dim'], architecture['cross_rank'])
                self.cross_layers = nn.ModuleList([nn.Sequential(nn.Linear(width, rank, bias=False), nn.Linear(rank, width, bias=True)) for _ in range(architecture['n_cross_layers'])])
                self.head = nn.Linear(width, 1)

            def forward(self, x):
                x0 = x
                for cross in self.cross_layers:
                    x = x0 * cross(x) + x
                return self.head(x)
        return LowRankCrossOnly()
    if architecture['family'] == 'torch_mlp':
        layers, width = ([], architecture['input_dim'])
        for hidden in architecture['hidden_dims']:
            layers.append(nn.Linear(width, hidden))
            if architecture['layer_norm']:
                layers.append(nn.LayerNorm(hidden))
            layers.extend([nn.ReLU(), nn.Dropout(architecture['dropout'])])
            width = hidden
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    class ResidualBlock(nn.Module):

        def __init__(self, hidden, expansion, dropout):
            super().__init__()
            self.body = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * expansion), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden * expansion, hidden), nn.Dropout(dropout))

        def forward(self, x):
            return x + self.body(x)

    class TabularResNet(nn.Module):

        def __init__(self):
            super().__init__()
            hidden = architecture['hidden_dim']
            self.input = nn.Linear(architecture['input_dim'], hidden)
            self.blocks = nn.Sequential(*[ResidualBlock(hidden, architecture['expansion_factor'], architecture['dropout']) for _ in range(architecture['n_blocks'])])
            self.output = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x):
            return self.output(self.blocks(self.input(x)))
    return TabularResNet()

def _raw_matrix(frame, columns, dtypes):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing neural input columns: {sorted(absent)}')
    if not frame.columns.is_unique:
        raise ValueError('Duplicate input columns')
    # Restore saved source precision first, then preprocess in float64 before
    # the explicitly recorded float32 neural matrix conversion.
    values = frame.loc[:, columns].astype(dtypes).to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values

def _fit_preprocessing(train, columns, dtypes, config):
    values = _raw_matrix(train, columns, dtypes)
    frame = pd.DataFrame(values, columns=columns)
    bounds = None
    if config.get('feature_clip_quantiles') is not None:
        lower, upper = map(float, config['feature_clip_quantiles'])
        if not 0 <= lower < upper <= 1:
            raise ValueError('feature_clip_quantiles requires 0<=lower<upper<=1')
        quantiles = frame.quantile([lower, upper]).fillna(0)
        lo, hi = (quantiles.iloc[0].to_numpy(), quantiles.iloc[1].to_numpy())
        values = np.clip(values, lo, hi)
        bounds = {'lower': lo.tolist(), 'upper': hi.tolist(), 'quantiles': [lower, upper]}
    imputation = pd.DataFrame(values, columns=columns).median().fillna(0).to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, imputation)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    scale = np.where(std > 1e-12, std, 1.0)
    standardize = config.get('standardize', True)
    if type(standardize) is not bool:
        raise ValueError('standardize must be boolean')
    return {'fillna': dict(zip(columns, imputation.tolist())), 'fill_strategy': 'training_median; all-missing=0', 'standardize': standardize, 'mean': mean.tolist(), 'scale': scale.tolist(), 'clip_bounds': bounds, 'statistics_fit_only_after_label_cutoff': True, 'matrix_dtype': 'float32', 'preprocessing_dtype': 'float64'}

def _matrix(frame, metadata):
    values = _raw_matrix(frame, metadata['features'], metadata['feature_dtypes'])
    prep = metadata['preprocessing']
    if prep['clip_bounds'] is not None:
        values = np.clip(values, prep['clip_bounds']['lower'], prep['clip_bounds']['upper'])
    fill = np.array([prep['fillna'][c] for c in metadata['features']], dtype=float)
    values = np.where(np.isfinite(values), values, fill)
    if prep['standardize']:
        values = (values - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite or overflowing float32 neural inputs after frozen preprocessing')
    return result

@dataclass
class FittedModel:
    estimator: object
    meta: dict

    @property
    def columns(self):
        return self.meta['features']

    @property
    def metadata(self):
        return self.meta

    def predict(self, frame, *, batch_size=None):
        """Float64, quantized raw scores in caller row order, no labels needed.

        batch_size is a memory/preprocessing chunk hint. Neural execution always
        pads to the frozen kernel batch size to reduce call-size differences.
        It cannot guarantee bit equality across hardware/library versions.
        """
        if self.meta['parameters'].get('predict_by_date', False):
            if not set(KEY).issubset(frame) or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
                raise ValueError('predict_by_date requires unique nonmissing 日期/代码 keys')
            order = frame[KEY].copy()
            order['_row_position'] = np.arange(len(frame))
            order = order.sort_values(KEY, kind='stable')
            result = np.empty(len(frame), dtype=np.float64)
            for _, part in order.groupby('日期', sort=False):
                positions = part['_row_position'].to_numpy()
                result[positions] = self._predict_rows(frame.iloc[positions], batch_size=batch_size)
            return result
        return self._predict_rows(frame, batch_size=batch_size)

    def _predict_rows(self, frame, *, batch_size=None):
        """Original row-order kernel; unchanged arithmetic for pre-existing models."""
        torch = _torch()
        parameters = self.meta['parameters']
        requested = parameters['predict_batch_size'] if batch_size is None else _positive_int(batch_size, 'batch_size')
        kernel = parameters['inference_kernel_batch_size']
        # Align preprocessing chunks to complete kernel blocks; arbitrary outer
        # batch hints therefore never change neural matrix multiplication shape.
        chunk = max(kernel, requested // kernel * kernel)
        result = np.empty(len(frame), dtype=np.float64)
        self.estimator.eval()
        with threadpool_limits(limits=2), torch.inference_mode():
            for start in range(0, len(frame), chunk):
                matrix = _matrix(frame.iloc[start:start + chunk], self.meta)
                for offset in range(0, len(matrix), kernel):
                    actual = matrix[offset:offset + kernel]
                    padded = np.zeros((kernel, len(self.columns)), dtype=np.float32)
                    padded[:len(actual)] = actual
                    if self.meta['family'] == 'torch_tabm':
                        heads = self.estimator(torch.from_numpy(padded))
                        if heads.shape != (kernel, parameters['k'], 1):
                            raise ValueError('TabM must return (batch,k,1) regression heads')
                        output = heads.squeeze(-1).mean(dim=1).cpu().numpy()
                    else:
                        output = self.estimator(torch.from_numpy(padded)).reshape(-1).cpu().numpy()
                    result[start + offset:start + offset + len(actual)] = output[:len(actual)]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite neural prediction')
        return np.round(result, parameters['raw_score_round_decimals'])

    def save(self, directory):
        return save(self, directory)
# Named alias for callers that prefer a distinctive estimator wrapper.
DeepFittedModel = FittedModel

def fit(train, config, *, year, groups=None, columns=None):
    """One fixed-epoch training run; caller owns its experimental reservation."""
    started = time.perf_counter()
    config = deepcopy(config)
    family = config.get('model')
    if family not in FAMILIES:
        raise ValueError('deep_learner supports torch_mlp/torch_resnet/torch_tabm/torch_dcn')
    year = int(year)
    cutoff = year * 10000 + 101
    cols = resolve_features(config, groups=groups, columns=columns)
    selected = train['日期'].ge(int(config.get('train_start', 20160101))) & train.exit_date.lt(cutoff) & np.isfinite(train.future_return)
    data = train.loc[selected].sort_values(KEY, kind='stable').reset_index(drop=True)
    if data.empty or data.duplicated(KEY).any():
        raise ValueError('Neural training rows empty or duplicate')
    if data.exit_date.le(data['日期']).any() or data.exit_date.max() >= cutoff:
        raise ValueError('Neural training label boundaries invalid')
    if config.get('precision', 'preserve') not in {'preserve', 'float32', 'float64'}:
        raise ValueError('precision must be preserve/float32/float64')
    precision = config.get('precision', 'preserve')
    if any((not pd.api.types.is_numeric_dtype(data[c]) for c in cols)):
        raise ValueError('Neural feature columns must be numeric')
    dtypes = {c: str(data[c].dtype) if precision == 'preserve' else precision for c in cols}
    parameters, architecture = _parameters(config, len(cols))
    preprocessing = _fit_preprocessing(data, cols, dtypes, config)
    meta = dict(format_version=1, deep_format_version=1, family=family, config=config, features=cols, feature_dtypes=dtypes, feature_precision=precision, preprocessing=preprocessing, architecture=architecture, parameters=parameters, prediction_dtype='float64 (quantized from float32 inference)', evaluation_year=year, train_label_cutoff_exclusive=cutoff, train_rows=len(data), train_signal_start=int(data['日期'].min()), train_signal_end=int(data['日期'].max()), train_label_exit_max=int(data.exit_date.max()), label='O(t+2)/O(t+1)-1', horizon=1, sample_weight='inverse training-day stock count, normalized mean 1; optional calendar half-life', validation_split=None, early_stopping=False, device='cpu', n_jobs=2)
    values = _matrix(data, meta)
    if family == 'torch_tabm' and architecture['num_embeddings'] is not None:
        meta['numerical_embedding'] = _fit_embedding_bins(values, parameters)
    transformed, target_meta = target_values(data, config)
    targets = np.asarray(transformed, dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError('Transformed neural target overflows float32')
    weights = 1.0 / data.groupby('日期')['代码'].transform('size').to_numpy(dtype=float)
    weights /= weights.mean()
    if config.get('half_life') is not None:
        half_life = float(config['half_life'])
        if not np.isfinite(half_life) or half_life <= 0:
            raise ValueError('half_life must be finite and positive')
        dates = pd.to_datetime(data['日期'].astype(str))
        weights *= np.exp2(-(dates.max() - dates).dt.days.to_numpy() / half_life)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Invalid neural sample weights')
    torch = _torch()
    seed = parameters['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    estimator = _network(torch, architecture, meta.get('numerical_embedding')).to(device='cpu', dtype=torch.float32)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'], foreach=False)
    x = torch.from_numpy(values)
    y = torch.from_numpy(targets)
    w = torch.from_numpy(weights.astype(np.float32))
    rng = np.random.default_rng(seed)
    history = []
    with threadpool_limits(limits=2):
        for epoch in range(parameters['epochs']):
            epoch_started = time.perf_counter()
            estimator.train()
            permutation = rng.permutation(len(data))
            accumulated = 0.0
            for begin in range(0, len(data), parameters['batch_size']):
                index = torch.from_numpy(permutation[begin:begin + parameters['batch_size']])
                optimizer.zero_grad(set_to_none=True)
                if family == 'torch_tabm':
                    prediction = estimator(x[index])
                    if prediction.shape != (len(index), parameters['k'], 1):
                        raise ValueError('TabM training output must be (batch,k,1)')
                    # Crucial: independent member losses, THEN average losses.
                    # Computing squared error of average(predictions) is wrong.
                    member_errors = (prediction.squeeze(-1) - y[index, None]).square()
                    loss = (w[index] * member_errors.mean(dim=1)).mean()
                else:
                    prediction = estimator(x[index]).reshape(-1)
                    # Mean rather than division by this batch's weight sum preserves
                    # the equal-date objective under uniform shuffled row sampling.
                    loss = (w[index] * (prediction - y[index]).square()).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite neural training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), parameters['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                accumulated += float(loss.detach()) * len(index)
            history.append({'epoch': epoch + 1, 'weighted_online_training_mse': accumulated / len(data), 'seconds': time.perf_counter() - epoch_started})
    estimator.eval()
    if any((not torch.isfinite(parameter).all().item() for parameter in estimator.parameters())):
        raise FloatingPointError('Nonfinite fitted neural weights')
    meta.update(target_transform=target_meta, train_history=history, parameter_count=sum((p.numel() for p in estimator.parameters())), fit_seconds=time.perf_counter() - started, learner_sha256=_hash(Path(__file__)), target_learner_sha256=_hash(Path(__file__).resolve().parent / 'learner.py'), versions={'torch': torch.__version__, **{p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'threadpoolctl')}}, determinism={'seed': seed, 'cpu_threads': torch.get_num_threads(), 'deterministic_algorithms': True, 'training_shuffle': 'seeded yearly training rows only', 'inference_fixed_kernel_batch': parameters['inference_kernel_batch_size'], 'canonical_prediction_by_date': parameters['predict_by_date'], 'raw_score_round_decimals': parameters['raw_score_round_decimals'], 'cross_platform_bitwise_guarantee': False})
    if family == 'torch_tabm':
        tabm = _tabm()
        meta['versions'].update(tabm=tabm.__version__, rtdl_num_embeddings=importlib.metadata.version('rtdl_num_embeddings'))
        meta.update(ensemble_size=parameters['k'], training_loss='mean_over_rows(weight * mean_over_heads(squared_error))', inference_aggregation='mean_regression_heads_before_frozen_raw_rounding', official_tabm_source_sha256=_hash(tabm.__file__))
    if family == 'torch_dcn':
        meta.update(architecture_description='Low-rank DCNv2-style cross-only network plus linear head; not full paper MoE', cross_equation='x_next = x0 * (U(V_transpose(x)) + bias) + x', training_loss='mean_over_rows(weight * squared_error)')
    return FittedModel(estimator, meta)

def predict(model, frame):
    return model.predict(frame)

def save(model, directory):
    torch = _torch()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'model_meta.json').exists() or (directory / 'model.pt').exists():
        raise FileExistsError('Refusing to overwrite a neural model artifact')
    artifact = directory / 'model.pt'
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.estimator.state_dict().items()}
    torch.save({'state_dict': state}, artifact)
    model.meta.update(artifact=artifact.name, artifact_sha256=_hash(artifact), artifact_format='torch_state_dict_weights_only')
    _json(directory / 'model_meta.json', model.meta)
    return directory

def load(directory):
    torch = _torch()
    directory = Path(directory).resolve()
    meta = json.loads((directory / 'model_meta.json').read_text(encoding='utf-8'))
    if meta.get('format_version') != 1 or meta.get('deep_format_version') != 1 or meta.get('family') not in FAMILIES:
        raise ValueError('Unsupported neural model metadata')
    artifact = (directory / meta['artifact']).resolve()
    if artifact.parent != directory or _hash(artifact) != meta['artifact_sha256']:
        raise ValueError('Neural artifact path/checksum invalid')
    if meta['architecture']['input_dim'] != len(meta['features']):
        raise ValueError('Neural input width differs from saved feature schema')
    parameters, architecture = _parameters({'model': meta['family'], 'params': meta['parameters']}, len(meta['features']))
    if architecture != meta['architecture']:
        raise ValueError('Neural architecture/parameters mismatch')
    estimator = _network(torch, architecture, meta.get('numerical_embedding')).to(device='cpu', dtype=torch.float32)
    payload = torch.load(artifact, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'state_dict'}:
        raise ValueError('Invalid neural tensor payload')
    if meta['family'] == 'torch_tabm' and architecture['num_embeddings'] is not None:
        for name, buffer in estimator.named_buffers():
            if name not in payload['state_dict'] or not torch.equal(buffer, payload['state_dict'][name]):
                raise ValueError('Numerical embedding metadata boundaries differ from frozen state buffers')
    estimator.load_state_dict(payload['state_dict'], strict=True)
    estimator.eval()
    if any((not torch.isfinite(p).all().item() for p in estimator.parameters())):
        raise ValueError('Nonfinite loaded neural weights')
    return FittedModel(estimator, meta)
