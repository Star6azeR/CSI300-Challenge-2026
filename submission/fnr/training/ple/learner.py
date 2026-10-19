"""Training-only label transforms and portable estimators for the local search.

No experiment registration, evaluation, data loading, or feature construction occurs
here. The caller owns those operations. See LEARNER.txt for the API contract.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
KEY = ['日期', '代码']
M5_PARAMS = dict(n_estimators=80, learning_rate=0.04, num_leaves=15, max_depth=5, min_child_samples=300, reg_lambda=10, colsample_bytree=0.85, subsample=0.8, subsample_freq=1, random_state=42, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
M5_CONFIG = dict(model='lgbm', features='extended', target='rank', rank_center=0.0, train_start=20160101, precision='preserve', params=deepcopy(M5_PARAMS))
B0_CONFIG = dict(model='ridge', features='B0', target='rank', rank_center=0.5, train_start=20160101, precision='preserve', params={'alpha': 1000.0}, fillna=0.0, standardize=False)
R43_CONFIG = dict(model='ridge', features='base', target='rank', rank_center=0.0, train_start=20160101, precision='preserve', params={'alpha': 100.0}, fillna=0.5, standardize=False)
FORBIDDEN = set(KEY + ['entry_date', 'exit_date', 'future_return', 'target', 'entry_open', 'exit_open', 'training_industry', '行业代码'])

def resolve_features(config, groups=None, columns=None):
    """Ordered union of named groups, or explicit columns; labels are rejected."""
    if columns is None:
        columns = config.get('columns')
    if columns is None:
        if groups is None:
            raise ValueError('Supply groups or explicit columns')
        names = config.get('features', 'extended')
        names = [names] if isinstance(names, str) else list(names)
        aliases = {'extended63': 'extended', 'base43': 'base', 'barra10': 'barra', 'B0': 'b0', 'b0': 'B0'}
        columns = []
        for name in names:
            actual = name if name in groups else aliases.get(name, name)
            if actual not in groups:
                raise KeyError(f'Unknown feature group {name!r}')
            columns.extend(groups[actual])
    columns = list(dict.fromkeys(columns))
    excluded = config.get('exclude', [])
    columns = [c for c in columns if not any((term in c for term in excluded))]
    if not columns or FORBIDDEN.intersection(columns):
        raise ValueError('Features must be nonempty and exclude labels/key/industry fields')
    return columns

def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def target_values(train, config):
    """Transform only the already-cut training rows; never accepts test labels."""
    kind = config.get('target', 'rank')
    center = float(config.get('rank_center', 0.0))
    opts = config.get('target_params', {})
    day = train['日期']
    y = train['future_return'].astype('float64').copy()
    info = {'kind': kind, 'rank_center': center, 'parameters': deepcopy(opts)}
    if kind == 'rank':
        # The exact original M5 used cached pct ranks in (0, 1], without -.5.
        values = train['target'].to_numpy(dtype='float64') - center
    elif kind.lower() in {'rawmad', 'raw_mad'}:
        floor, clip = (float(opts.get('floor', 0.002)), float(opts.get('clip', 3.0)))
        if floor <= 0 or clip <= 0:
            raise ValueError('rawMAD floor and clip must be positive')
        median = y.groupby(day).transform('median')
        mad = (y - median).abs().groupby(day).transform('median')
        scale = (float(opts.get('mad_multiplier', 1.4826)) * mad).clip(lower=floor)
        values = ((y - median) / scale).clip(-clip, clip).to_numpy()
        info.update(formula='clip((r-daily_median)/(1.4826*daily_MAD floor), ±clip)', floor=floor, clip=clip, mad_multiplier=float(opts.get('mad_multiplier', 1.4826)))
    elif kind in {'residual_rank', 'industry_rank'}:
        if kind == 'residual_rank':
            bcols = list(opts.get('barra_columns', [c for c in train if c.startswith('barra_')]))
            if not bcols or FORBIDDEN.intersection(bcols):
                raise ValueError('residual_rank requires explicit or inferred Barra columns')
            for _, ix in train.groupby('日期', sort=False).groups.items():
                b = train.loc[ix, bcols].replace([np.inf, -np.inf], np.nan).fillna(0.5).to_numpy(dtype=float)
                x = np.column_stack([np.ones(len(ix)), b])
                y.loc[ix] -= x @ np.linalg.lstsq(x, y.loc[ix].to_numpy(), rcond=float(opts.get('rcond', 1e-06)))[0]
            info.update(barra_columns=bcols, rcond=float(opts.get('rcond', 1e-06)))
        else:
            field = opts.get('industry_column', 'training_industry')
            if field not in train:
                raise ValueError(f'industry_rank requires training-only {field!r}')
            industry = train[field].astype('string').str.strip()
            if industry.isna().any() or industry.eq('').any():
                raise ValueError('Missing training industry coverage')
            y -= y.groupby([day, industry]).transform('mean')
            info.update(industry_column=field, residualization='daily industry mean subtraction')
        values = y.groupby(day).rank(pct=True).to_numpy() - center
    elif kind == 'raw':
        clip = float(opts.get('clip', 0.1))
        values = y.clip(-clip, clip).to_numpy() * float(opts.get('scale', 10.0))
        info.update(clip=clip, scale=float(opts.get('scale', 10.0)))
    else:
        raise ValueError(f'Unsupported target {kind!r}')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite transformed training target')
    return (values, info)

def _matrix(frame, metadata):
    cols = metadata['features']
    missing = set(cols).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    # Saved per-column dtypes keep float32 and original float64 feature paths distinct.
    x = frame.loc[:, cols].astype(metadata['feature_dtypes']).replace([np.inf, -np.inf], np.nan)
    prep = metadata['preprocessing']
    if prep.get('clip_bounds') is not None:
        bounds = prep['clip_bounds']
        x = x.clip(lower=pd.Series(bounds['lower'], index=cols), upper=pd.Series(bounds['upper'], index=cols), axis=1)
    if prep['fillna'] is not None:
        x = x.fillna(prep['fillna'])
    if prep.get('standardize'):
        x = (x - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    return x

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

    def predict(self, frame):
        x = _matrix(frame, self.meta)
        if len(x) == 0:
            return np.empty(0, dtype=float)
        with threadpool_limits(limits=4):
            if self.meta['family'] == 'lgbm':
                scores = self.estimator.predict(x, num_threads=4)
            else:
                scores = x.to_numpy() @ self.estimator['coef'] + self.estimator['intercept']
        scores = np.asarray(scores, dtype='float64')
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError('Prediction is nonfinite or has the wrong shape')
        return scores

    def save(self, directory):
        return save(self, directory)

def fit(train, config, *, year, groups=None, columns=None):
    """Fit a frozen config, independently enforcing strict training label cutoff."""
    config = deepcopy(config)
    family = config.get('model', 'lgbm')
    if family not in {'lgbm', 'ridge', 'huber'}:
        raise ValueError(f'Unsupported model {family!r}')
    year = int(year)
    cutoff = year * 10000 + 101
    cols = resolve_features(config, groups, columns)
    train_start = int(config.get('train_start', 20160101))
    selected = train['日期'].ge(train_start) & train.exit_date.lt(cutoff) & np.isfinite(train.future_return)
    data = train.loc[selected].sort_values(KEY, kind='stable').reset_index(drop=True)
    if data.empty or data.duplicated(KEY).any():
        raise ValueError('Training rows are empty or have duplicate date/code keys')
    if data.exit_date.le(data['日期']).any():
        raise ValueError('Label exits must follow signal dates')
    precision = config.get('precision', 'preserve')
    if precision not in {'preserve', 'float64', 'float32'}:
        raise ValueError('precision must be preserve, float64, or float32')
    dtypes = {c: str(data[c].dtype) if precision == 'preserve' else precision for c in cols}
    if any((not pd.api.types.is_numeric_dtype(data[c]) for c in cols)):
        raise ValueError('All features must be numeric')
    fill_values = None
    if family != 'lgbm':
        requested = config.get('fillna', {c: 0.0 if c.startswith('v3_') else 0.5 for c in cols})
        if isinstance(requested, dict):
            if set(requested) != set(cols):
                raise ValueError('fillna mapping must name every feature exactly once')
            fill_values = {c: float(requested[c]) for c in cols}
        else:
            fill_values = float(requested)
        if not np.isfinite(list(fill_values.values()) if isinstance(fill_values, dict) else fill_values).all():
            raise ValueError('fillna values must be finite')
    prep = {'fillna': fill_values, 'standardize': bool(config.get('standardize', False)), 'clip_bounds': None}
    if family == 'lgbm' and prep['standardize']:
        raise ValueError('standardize is supported only for linear estimators')
    meta = dict(format_version=1, family=family, features=cols, config=config, feature_precision=precision, feature_dtypes=dtypes, preprocessing=prep, prediction_dtype='float64', evaluation_year=year, train_label_cutoff_exclusive=cutoff, train_rows=len(data), train_signal_start=int(data['日期'].min()), train_signal_end=int(data['日期'].max()), train_label_exit_max=int(data.exit_date.max()), label='O(t+2)/O(t+1)-1', horizon=1)
    quantiles = config.get('feature_clip_quantiles')
    if quantiles is not None:
        lo, hi = map(float, quantiles)
        if not 0 <= lo < hi <= 1:
            raise ValueError('feature_clip_quantiles requires 0 <= low < high <= 1')
        stats = data[cols].replace([np.inf, -np.inf], np.nan).quantile([lo, hi]).fillna(0.5)
        prep['clip_bounds'] = {'lower': stats.iloc[0].tolist(), 'upper': stats.iloc[1].tolist()}
        prep['clip_quantiles'] = [lo, hi]
    if prep['standardize']:
        prep['standardize'] = False
        x = _matrix(data, meta)
        prep.update(mean=x.mean().to_numpy(dtype=float).tolist(), scale=x.std(ddof=0).where(lambda z: z.gt(1e-12), 1.0).to_numpy(dtype=float).tolist(), standardize=True)
    x = _matrix(data, meta)
    y, target_meta = target_values(data, config)
    weights = 1 / data.groupby('日期')['代码'].transform('size')
    weights /= weights.mean()
    if config.get('half_life') is not None:
        half_life = float(config['half_life'])
        if half_life <= 0:
            raise ValueError('half_life must be positive')
        dates = pd.to_datetime(data['日期'].astype(str))
        weights *= np.exp2(-(dates.max() - dates).dt.days / half_life)
    with threadpool_limits(limits=4):
        if family == 'lgbm':
            import lightgbm as lgb
            parameters = deepcopy(M5_PARAMS)
            parameters.update(config.get('params', {}))
            # Reject alias bypasses of the four-thread ceiling and GPU execution.
            for key in ('n_jobs', 'num_threads', 'num_thread', 'nthread', 'nthreads'):
                if key in parameters and (int(parameters[key]) < 1 or int(parameters[key]) > 4):
                    raise ValueError(f'{key} must be in [1,4]')
            if parameters.get('device_type', parameters.get('device', 'cpu')) != 'cpu':
                raise ValueError('Only CPU estimators are allowed')
            estimator = lgb.LGBMRegressor(**parameters)
            estimator.fit(x, y, sample_weight=weights)
            estimator = estimator.booster_
        else:
            from sklearn.linear_model import HuberRegressor, Ridge
            parameters = {'alpha': config.get('alpha', 100.0)} if family == 'ridge' else {'alpha': 0.01, 'epsilon': 1.35, 'max_iter': 500, 'tol': 1e-05}
            parameters.update(config.get('params', {}))
            estimator = (Ridge if family == 'ridge' else HuberRegressor)(**parameters)
            estimator.fit(x, y, sample_weight=weights)
            meta['linear_fit_n_iter'] = int(estimator.n_iter_) if 'n_iter_' in vars(estimator) and np.isscalar(estimator.n_iter_) else None
            estimator = {'coef': np.asarray(estimator.coef_, dtype=float), 'intercept': float(estimator.intercept_)}
    meta.update(parameters=parameters, target_transform=target_meta, sample_weight='equal day weight, normalized mean 1; optional calendar-day half-life', learner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'threadpoolctl')})
    return FittedModel(estimator, meta)

def predict(model, frame):
    return model.predict(frame)

def save(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'model_meta.json').exists():
        raise FileExistsError('Refusing to overwrite an existing model package')
    if model.meta['family'] == 'lgbm':
        artifact = directory / 'model.txt'
        model.estimator.save_model(str(artifact))
        pd.DataFrame({'feature': model.columns, 'gain': model.estimator.feature_importance('gain'), 'split': model.estimator.feature_importance('split')}).to_csv(directory / 'importance.csv', index=False)
    else:
        artifact = directory / 'model.npz'
        np.savez(artifact, **model.estimator)
    model.meta['artifact'] = artifact.name
    model.meta['artifact_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _json(directory / 'model_meta.json', model.meta)
    return directory

def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'model_meta.json').read_text(encoding='utf-8'))
    if meta.get('format_version') != 1:
        raise ValueError('Unsupported learner model format')
    artifact = directory / meta['artifact']
    if artifact.parent.resolve() != directory.resolve():
        raise ValueError('Artifact must be directly inside the package directory')
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != meta['artifact_sha256']:
        raise ValueError('Saved model checksum mismatch')
    if meta['family'] == 'lgbm':
        import lightgbm as lgb
        estimator = lgb.Booster(model_file=str(artifact))
        if estimator.feature_name() != meta['features']:
            raise ValueError('Booster feature order differs from metadata')
    elif meta['family'] in {'ridge', 'huber'}:
        with np.load(artifact, allow_pickle=False) as data:
            estimator = {'coef': data['coef'].copy(), 'intercept': float(data['intercept'])}
        if estimator['coef'].shape != (len(meta['features']),):
            raise ValueError('Linear coefficient count differs from features')
    else:
        raise ValueError('Unknown saved model family')
    return FittedModel(estimator, meta)
