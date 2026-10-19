"""Frozen ensemble inference over the bounded daily feature pipeline."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..features.bounded import build_features
from .kernels import read_json, sha256, neural_matrix, load_lgbm, load_tabm, predict_lgbm, predict_tabm
KEY = ['日期', '代码']
STYLES = ['市值', '贝塔', '动量', '残差波动', '非线性市值', '账面市值比', '流动性', '盈利', '成长', '杠杆']
TABLES = ('Factor33_winsor', 'Barra_factor', 'MarketData')
FORBIDDEN = {'future_return', 'target', 'entry_date', 'exit_date', 'entry_open', 'exit_open', 'score', 'raw_score'}

def rank(values):
    return pd.Series(values).rank(method='average', pct=True).to_numpy()

def required_lookback(family, params):
    lag = 0
    if family in ('two_day_smoothing', 'disagreement_gated_smoothing', 'finite_rank_hysteresis', 'fresh_signal_protected_smoothing'):
        lag = 1
    if family == 'three_day_smoothing':
        lag = 2
    post = params.get('postprocess')
    if post:
        kind = post['kind']
        if kind in ('global_smoothing', 'middle_smoothing', 'middle_then_barra'):
            lag = max(lag, 1)
        elif kind == 'middle_reversal_smoothing':
            lag = 2
        elif kind not in ('barra_residualization', 'volatility_shrinkage'):
            raise ValueError('Unsupported postprocess ' + kind)
    return 61 + lag

class InferenceModel:

    def __init__(self, model_dir):
        directory = Path(model_dir).resolve()
        manifest = read_json(directory / 'manifest.json')
        expected_files = {str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file() and f.name != 'manifest.json'}
        if set(manifest['files_sha256']) != expected_files:
            raise ValueError('Model manifest does not cover every model file')
        for relative, expected in manifest['files_sha256'].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
                raise ValueError('Model checksum/path mismatch: ' + relative)
        self.meta = read_json(directory / 'ensemble.json')
        self.family = self.meta['family']
        self.params = self.meta['parameters']
        self.lookback = required_lookback(self.family, self.params)
        if self.lookback != self.meta['lookback'] or self.lookback > 63:
            raise ValueError('Invalid bounded history contract')
        self.lgb, self.lgb_meta = load_lgbm(directory / 'lgbm')
        self.tab, self.tab_meta = load_tabm(directory / 'tabm')
        self.build_features = build_features
        self.columns = list(self.meta['features'])
        self.components = self.params.get('components', []) if self.family == 'auxiliary_stack' else [self.params] if self.family == 'cached_auxiliary_blend' else []
        self.aux = []
        for component in self.components:
            state, metadata = load_tabm(directory / component['model_name'])
            self.aux.append((float(component['weight']), state, metadata))
        if not 0 <= sum((x[0] for x in self.aux)) <= 0.5:
            raise ValueError('Invalid total auxiliary weight')
        for metadata in [x[2] for x in self.aux]:
            for key in ['features', 'feature_dtypes', 'preprocessing']:
                if metadata[key] != self.tab_meta[key]:
                    raise ValueError('Shared neural preprocessing contract mismatch')
        for metadata in [self.lgb_meta, self.tab_meta, *[x[2] for x in self.aux]]:
            if metadata['features'] != self.columns or len(self.columns) != 347:
                raise ValueError('Component feature schema mismatch')
            if int(metadata['train_label_exit_max']) >= 20230101:
                raise ValueError('Frozen training label cutoff mismatch')

    def _scores(self, features):
        prepared = neural_matrix(features, self.tab_meta)
        a = rank(predict_lgbm(self.lgb, features, self.lgb_meta))
        b = rank(predict_tabm(self.tab, features, self.tab_meta, prepared))
        # Native frozen leaf scores are round12 daily percent ranks, followed by blend component rerank.
        a = rank(np.round(a, 12))
        b = rank(np.round(b, 12))
        base = np.round(0.5 * (a + b), 12)
        q = rank(base)
        total = sum((x[0] for x in self.aux))
        stack = (1 - total) * q
        for w, state, meta in self.aux:
            if not 0 <= w <= 0.35:
                raise ValueError('Auxiliary component weight outside bounds')
            aux = rank(np.round(rank(predict_tabm(state, features, meta, prepared)), 12))
            stack += w * aux
        return {'base': base, 'q': q, 'lgb': a, 'tab': b, 'stack': stack}

    def predict(self, as_of_date, data, *, return_features=False):
        day = int(pd.Timestamp(as_of_date).strftime('%Y%m%d'))
        if set(data) != set(TABLES):
            raise ValueError('Exactly the three allowed raw input tables required')
        for name, t in data.items():
            if not t.columns.is_unique or not set(KEY).issubset(t) or t.duplicated(KEY).any() or FORBIDDEN.intersection(t.columns) or t['日期'].gt(day).any():
                raise ValueError('Illegal raw input ' + name)
        dates = np.sort(data['MarketData']['日期'].unique())[-self.lookback:]
        if not len(dates) or dates[-1] != day:
            raise ValueError('Market calendar does not include as_of_date')
        data = {name: table.loc[table['日期'].isin(dates)].copy() for name, table in data.items()}
        # Retain the complete bounded prefix for identical rolling arithmetic.
        # target_date emits/ranks only that date's constituent rows; the whole
        # call still consumes at most the configured 61/62/63 exchange dates.
        lag = self.lookback - 61
        per = []
        # One shared trailing calculation; output ranking remains per date.
        features, _ = self.build_features(data['Factor33_winsor'], data['Barra_factor'], data['MarketData'], calendar=dates, target_dates=dates[-(lag + 1):])
        for d in dates[-(lag + 1):]:
            f = features.loc[features['日期'].eq(d), KEY + self.columns].sort_values(KEY).reset_index(drop=True)
            expected = data['Barra_factor'].loc[data['Barra_factor']['日期'].eq(d), KEY].sort_values(KEY).reset_index(drop=True)
            if not np.array_equal(f[KEY].to_numpy(), expected.to_numpy()):
                raise ValueError('Exact historical Barra coverage failed')
            s = self._scores(f)
            per.append((f['代码'].tolist(), s))
        codes = per[-1][0]
        cur = per[-1][1]

        def prior(name, k, fallback):
            if k >= len(per):
                return fallback.copy()
            values = pd.Series(per[-1 - k][1][name], index=per[-1 - k][0]).reindex(codes).to_numpy()
            return np.where(np.isfinite(values), values, fallback)
        current = cur['q']
        previous = prior('q', 1, current)
        family = self.family
        params = self.params
        if family == 'baseline':
            score = cur['base'].copy()
        elif family in ('auxiliary_stack', 'cached_auxiliary_blend'):
            score = cur['stack'].copy()
        elif family == 'component_reweight':
            score = cur['base'] + (float(params['tabm_weight']) - 0.5) * (cur['tab'] - cur['lgb'])
        elif family in ('two_day_smoothing', 'disagreement_gated_smoothing'):
            w = float(params['weight']) if family == 'two_day_smoothing' else float(params['weight_max']) * np.clip(2 * np.abs(cur['lgb'] - cur['tab']), 0, 1)
            score = (1 - w) * current + w * previous
        elif family == 'three_day_smoothing':
            weights = np.asarray(params['weights'], float)
            if weights.shape != (3,) or (weights < 0).any() or abs(weights.sum() - 1) > 1e-12:
                raise ValueError('Invalid three-day weights')
            # Exploration ignores absent second lag and renormalizes, rather than filling it.
            prev2 = prior('q', 2, current)
            values = np.vstack([current, previous, prev2])
            valid = np.isfinite(values)
            score = (np.where(valid, values, 0) * weights[:, None]).sum(0) / (valid * weights[:, None]).sum(0)
        elif family == 'finite_rank_hysteresis':
            delta = current - previous
            score = previous + np.sign(delta) * np.maximum(abs(delta) - float(params['delta']), 0)
        elif family == 'fresh_signal_protected_smoothing':
            limit = float(params['fresh_delta'])
            gate = np.clip(1 - np.abs(current - previous) / limit, 0, 1)
            w = float(params['weight']) * gate
            score = (1 - w) * current + w * previous
        elif family == 'partial_score_residualization':
            score = self._risk(current, data, day, codes, float(params['eta']), float(params.get('ridge_penalty_per_stock', 0.001))) - current.mean()
        else:
            raise AssertionError(family)
        post = params.get('postprocess')
        if post:
            q = rank(score)
            kind = post['kind']
            if kind == 'barra_residualization':
                score = self._risk(q, data, day, codes, float(post['eta']), 0.001)
            elif kind == 'volatility_shrinkage':
                gamma = float(post['gamma'])
                if not 0 <= gamma <= 0.2:
                    raise ValueError('Volatility shrinkage gamma out of bounds')
                exposure = data['Barra_factor'].loc[data['Barra_factor']['日期'].eq(day)].set_index('代码').reindex(codes)['残差波动'].to_numpy(float)
                risk = pd.Series(exposure).rank(method='average', pct=True).fillna(0.5).to_numpy()
                gate = np.clip((risk - 0.7) / 0.3, 0, 1)
                score = 0.5 + (q - 0.5) * (1 - gamma * gate)
            else:
                # Postprocessing previous values are ranks of historical PRE-postprocess auxiliary blends.
                name = 'stack' if family in ('auxiliary_stack', 'cached_auxiliary_blend') else 'q'
                prev = prior(name, 1, q)
                if len(per) > 1:
                    pv = rank(per[-2][1][name])
                    prev = pd.Series(pv, index=per[-2][0]).reindex(codes).to_numpy()
                    prev = np.where(np.isfinite(prev), prev, q)
                gate = np.ones(len(codes))
                if kind in ('middle_smoothing', 'middle_reversal_smoothing', 'middle_then_barra'):
                    h = lambda z: np.clip(1 - np.abs(z - 0.5) / 0.25, 0, 1)
                    gate = h(q) * h(prev)
                    if kind == 'middle_reversal_smoothing':
                        prev2 = q.copy()
                        if len(per) > 2:
                            pv2 = rank(per[-3][1][name])
                            prev2 = pd.Series(pv2, index=per[-3][0]).reindex(codes).to_numpy()
                            prev2 = np.where(np.isfinite(prev2), prev2, q)
                        d = q - prev
                        p = prev - prev2
                        gate *= (d * p < 0) & (abs(d) < 0.08) & (abs(p) < 0.08) & (abs(cur['lgb'] - cur['tab']) > 0.25)
                elif kind != 'global_smoothing':
                    raise ValueError('Unsupported post kind ' + kind)
                weight = float(post['weight'])
                if not 0 <= weight <= 0.35:
                    raise ValueError('Postprocess weight outside bounds')
                score = q + weight * gate * (prev - q)
                if kind == 'middle_then_barra':
                    score = self._risk(rank(score), data, day, codes, float(post['eta']), 0.001)
        score = np.round(score, 12)
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite final signal')
        result = pd.DataFrame({'date': pd.Timestamp(as_of_date).strftime('%Y-%m-%d'), 'code': codes, 'score': score})
        return (result, f) if return_features else result

    @staticmethod
    def _risk(q, data, day, codes, eta, ridge):
        if not 0 <= eta <= 0.5 or ridge < 0:
            raise ValueError('Risk control bounds')
        x = data['Barra_factor'].loc[data['Barra_factor']['日期'].eq(day)].set_index('代码').reindex(codes)[STYLES].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite current Barra styles')
        std = x.std(0)
        x = (x - x.mean(0)) / np.where(std > 1e-12, std, 1)
        beta = np.linalg.solve(x.T @ x + len(x) * ridge * np.eye(10), x.T @ (q - q.mean()))
        return q - eta * (x @ beta)
