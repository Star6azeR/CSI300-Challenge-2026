"""Daily asof membership and separate supervised-label masks."""
import numpy as np
import pandas as pd
KEY = ['日期', '代码']
HISTORY_SOURCE_LOOKBACK = 21
HISTORY_NAMES = ['raw_hist_ret_1', 'raw_hist_intraday', 'raw_hist_overnight', 'raw_hist_range_ratio', 'raw_hist_close_location', 'raw_hist_amount_ratio_5', 'raw_hist_turn_ratio_20', 'raw_hist_vol_20', 'raw_px_open_ret_1', 'raw_liq_log_amount', 'raw_px_upper_shadow', 'raw_px_lower_shadow']

def fit_preprocessing(frame, cols):
    """Two-pass float64 statistics over all asof rows, finalized as float32."""
    sums = np.zeros(len(cols))
    counts = np.zeros(len(cols))
    squares = np.zeros(len(cols))
    for start in range(0, len(frame), 8192):
        raw = frame.iloc[start:start + 8192][cols].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw)
        sums += np.where(valid, raw, 0).sum(0)
        counts += valid.sum(0)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    for start in range(0, len(frame), 8192):
        raw = frame.iloc[start:start + 8192][cols].to_numpy(dtype=np.float64)
        valid = np.isfinite(raw)
        squares += np.where(valid, (raw - mean) ** 2, 0).sum(0)
    std = np.sqrt(np.divide(squares, counts, out=np.zeros_like(squares), where=counts > 0))
    mean = np.nan_to_num(mean).astype('float32')
    std = np.where(np.isfinite(std) & (std > 1e-06), std, 1).astype('float32')
    return {'features': cols, 'mean': mean.tolist(), 'std': std.tolist(), 'clip': 8, 'history_names': HISTORY_NAMES}

def transform(frame, preprocessing):
    x = frame[preprocessing['features']].to_numpy(dtype=np.float32)
    if preprocessing.get('fillna_raw') is not None:
        fill = np.asarray(preprocessing['fillna_raw'], dtype=np.float32)
        x = np.where(np.isfinite(x), x, fill)
    x = (x - np.asarray(preprocessing['mean'], dtype=np.float32)) / np.asarray(preprocessing['std'], dtype=np.float32)
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    clip = preprocessing.get('clip', 8)
    return x if clip is None else np.clip(x, -clip, clip)

class DailyData:

    def __init__(self, frame, preprocessing, history_length=0):
        self.frame = frame[KEY].copy()
        self.days = np.sort(frame['日期'].unique())
        self.groups = list(frame.groupby('日期', sort=True).indices.values())
        self.width = max(map(len, self.groups))
        self.dim = len(preprocessing['features'])
        self.history_length = history_length
        self.x = np.zeros((len(self.groups), self.width, self.dim), dtype='float32')
        self.mask = np.zeros(self.x.shape[:2], dtype=bool)
        self.barra = np.zeros((len(self.groups), self.width, 10), dtype='float32')
        self.float_cap = np.zeros(self.mask.shape, dtype='float32')
        self.y_rank = np.zeros_like(self.mask, dtype='float32')
        self.y_cash = self.y_rank.copy()
        self.raw_return = self.y_rank.copy()
        self.label_mask = self.mask.copy()
        self.auxiliary_label_names = list(preprocessing.get('auxiliary_label_columns', []))
        if set(self.auxiliary_label_names) & set(preprocessing['features']):
            raise ValueError('Auxiliary future labels cannot be input features')
        if len(set(self.auxiliary_label_names)) != len(self.auxiliary_label_names):
            raise ValueError('Duplicate auxiliary label names')
        auxiliary_present = [name in frame for name in self.auxiliary_label_names]
        if any(auxiliary_present) and (not all(auxiliary_present)):
            raise ValueError('Partial auxiliary label columns')
        self.auxiliary_labels = np.zeros((*self.mask.shape, len(self.auxiliary_label_names)), dtype='float32')
        self.auxiliary_label_mask = np.zeros(self.auxiliary_labels.shape, dtype=bool)
        values = transform(frame, preprocessing)
        self.codes = []
        for i, pos in enumerate(self.groups):
            n = len(pos)
            self.x[i, :n] = values[pos]
            self.mask[i, :n] = True
            self.codes.append(frame.iloc[pos]['代码'].to_numpy())
            self.barra[i, :n] = frame.iloc[pos][preprocessing['features'][:10]].to_numpy(dtype=np.float32)
            if 'raw_float_market_cap' in frame:
                self.float_cap[i, :n] = np.nan_to_num(frame.iloc[pos].raw_float_market_cap.to_numpy(dtype=np.float32), nan=0, posinf=0, neginf=0)
            if 'future_return' in frame:
                raw = frame.iloc[pos].future_return.to_numpy(float)
                valid = np.isfinite(raw)
                self.label_mask[i, :n] = valid
                if valid.any():
                    v = raw[valid]
                    m = np.median(v)
                    mad = np.median(np.abs(v - m))
                    scale = mad if mad > 1e-08 else max(np.std(v), 1e-08)
                    self.y_rank[i, :n][valid] = pd.Series(v).rank(pct=True).to_numpy() - 0.5
                    self.y_cash[i, :n][valid] = np.clip((v - m) / (3 * scale), -1, 1)
                    self.raw_return[i, :n][valid] = v
            if self.auxiliary_label_names and all(auxiliary_present):
                auxiliary = frame.iloc[pos][self.auxiliary_label_names].to_numpy(dtype=np.float32)
                auxiliary_valid = np.isfinite(auxiliary) & self.label_mask[i, :n, None]
                self.auxiliary_labels[i, :n] = np.where(auxiliary_valid, auxiliary, 0)
                self.auxiliary_label_mask[i, :n] = auxiliary_valid
        self.hindices = None
        if history_length:
            source_lookback = int(preprocessing.get('history_source_lookback', HISTORY_SOURCE_LOOKBACK))
            if max(61, source_lookback + history_length - 1) > 120:
                raise ValueError('Composed history exceeds120')
            segment_column = preprocessing.get('history_segment_column')
            segments = None
            if segment_column:
                if segment_column not in frame:
                    raise ValueError('Missing declared causal history segment')
                segments = []
                for pos in self.groups:
                    values_in_group = frame.iloc[pos][segment_column].unique()
                    if len(values_in_group) != 1:
                        raise ValueError('A date has mixed teacher-fold history segments')
                    segments.append(values_in_group[0])
            # Missing historical membership remains missing; no future membership backfill.
            self.hindices = np.full((len(self.groups), self.width, history_length), -1, dtype=np.int32)
            maps = [{code: i * self.width + j for j, code in enumerate(c)} for i, c in enumerate(self.codes)]
            for i, codes in enumerate(self.codes):
                for l in range(history_length):
                    t = i - history_length + 1 + l
                    if t >= 0:
                        self.hindices[i, :len(codes), l] = [maps[t].get(code, -1) for code in codes]
                    if t >= 0 and segments is not None and (segments[t] != segments[i]):
                        self.hindices[i, :, l] = -1
            self.hchannels = [preprocessing['features'].index(c) for c in preprocessing.get('history_names', HISTORY_NAMES)]
            self.hvalues = self.x[..., self.hchannels].reshape(-1, len(self.hchannels))

    def batch(self, indices, torch):
        ix = np.asarray(indices)
        arrays = {'x': self.x, 'mask': self.mask, 'y_rank': self.y_rank, 'y_cash': self.y_cash, 'label_mask': self.label_mask, 'raw_return': self.raw_return, 'barra': self.barra, 'float_cap': self.float_cap}
        pack = {k: torch.from_numpy(value[ix]) for k, value in arrays.items()}
        pack['dates'] = torch.from_numpy(self.days[ix].astype(np.int64))
        if self.auxiliary_label_names:
            pack['auxiliary_labels'] = torch.from_numpy(self.auxiliary_labels[ix])
            pack['auxiliary_label_mask'] = torch.from_numpy(self.auxiliary_label_mask[ix])
        pack['history'] = None
        if self.hindices is not None:
            h = self.hindices[ix]
            valid = h >= 0
            values = self.hvalues[np.maximum(h, 0)]
            values[~valid] = 0
            pack['history'] = {'values': torch.from_numpy(values), 'mask': torch.from_numpy(valid)}
        return pack

    def forecasts(self, scores):
        parts = []
        for i, pos in enumerate(self.groups):
            f = self.frame.iloc[pos].rename(columns={'日期': 'date', '代码': 'code'}).copy()
            f['score'] = scores[i, :len(pos)]
            parts.append(f)
        out = pd.concat(parts, ignore_index=True)
        out.date = pd.to_datetime(out.date.astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
        return out
