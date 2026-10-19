"""Bounded point-in-time features for the CSI300 H=1 score search.

Only same-day historical constituent rows from Barra_factor are used. The longest
calculation consumes exactly 120 exchange dates, never 120 *observations* of a
particular stock. No forward fills, unbounded EWM, labels, disk reads or writes.

build_features(..., target_date=YYYYMMDD) returns only that date and is useful for
120-day inference replay. Omitting target_date returns all supplied dates.
All returned feature columns are float32; NaNs are intentionally retained.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .rolling import rolling as _fast_rolling
KEY = ['日期', '代码']
LOOKBACK = 120
RELATIVE = ('sma_20', 'ema_20', 'macd_diff_12_26_9', 'macd_dea_12_26_9', 'macd_hist_12_26_9', 'atr_14')
FLAGS = ('is_suspend', 'is_st', 'is_new')

def build_features(factors, barra, market, calendar=None, *, target_date=None):
    """Return (feature frame, named feature groups).

    calendar, when supplied, must be the exchange-date calendar covering the
    input. Dates later than the newest supplied market date are discarded.
    target_date is optional and does not change a target row's calculation.
    Inputs are never mutated. Constituents are filtered separately on every date.
    """
    for name, frame in [('factors', factors), ('barra', barra), ('market', market)]:
        if frame.duplicated(KEY).any():
            raise ValueError(f'{name} contains duplicate date/code keys')
    if market.empty or barra.empty:
        raise ValueError('Non-empty market and barra inputs are required')
    if target_date is not None:
        # Enforce the same legal horizon even if a caller supplies full history.
        observed = np.sort(market.loc[market['日期'].le(target_date), '日期'].unique())
        visible = observed[-LOOKBACK:]
        factors = factors[factors['日期'].isin(visible)]
        barra = barra[barra['日期'].isin(visible)]
        market = market[market['日期'].isin(visible)]
        if calendar is not None:
            calendar = [d for d in calendar if d in set(visible)]
    dates = np.sort(market['日期'].unique() if calendar is None else np.asarray([d for d in calendar if market['日期'].min() <= d <= market['日期'].max()]))
    dates = pd.Index(np.unique(dates), name='日期')
    codes = pd.Index(sorted(barra['代码'].unique()), name='代码')
    if not pd.Index(barra['日期'].unique()).isin(dates).all():
        raise ValueError('calendar omits a constituent date')
    bcols = [c for c in barra if c not in KEY + ['名称']]
    fcols = [c for c in factors if c not in KEY + ['名称']]
    raw_base = barra[KEY + bcols].rename(columns={c: 'barra_' + c for c in bcols})
    raw_base = raw_base.merge(factors[KEY + fcols], on=KEY, how='left', validate='one_to_one')
    raw_base = raw_base.merge(market[KEY + ['close']], on=KEY, how='left', validate='one_to_one')
    for c in RELATIVE:
        raw_base[c] = raw_base[c] / raw_base['close'].where(raw_base['close'].gt(0))
    raw_base = raw_base.drop(columns='close').sort_values(KEY).reset_index(drop=True)
    basecols = [c for c in raw_base if c not in KEY]
    raw_base[basecols] = raw_base[basecols].replace([np.inf, -np.inf], np.nan)
    base = raw_base.copy()
    continuous = [c for c in fcols if c not in FLAGS]
    base[continuous] = base.groupby('日期', sort=False)[continuous].rank(pct=True, method='average')
    select = np.ones(len(base), dtype=bool) if target_date is None else base['日期'].eq(target_date).to_numpy()
    selected = base.loc[select, KEY]
    ri = dates.get_indexer(selected['日期'])
    ci = codes.get_indexer(selected['代码'])
    columns = {c: base.loc[select, c].to_numpy(dtype=np.float32) for c in basecols}
    rawcols = []
    for c in basecols:
        name = 'raw_' + c
        columns[name] = raw_base.loc[select, c].to_numpy(dtype=np.float32)
        rawcols.append(name)
    legal = barra[KEY].merge(market, on=KEY, how='left', validate='one_to_one')
    legal_f = barra[KEY].merge(factors, on=KEY, how='left', validate='one_to_one')

    def pivot(frame, col):
        return frame.pivot(index='日期', columns='代码', values=col).reindex(index=dates, columns=codes)
    member = pivot(barra[KEY].assign(_member=1), '_member').eq(1)
    suspended = pivot(legal, 'is_suspend')
    close = pivot(legal, 'close').where(suspended.eq(0))
    opening = pivot(legal, 'open').where(suspended.eq(0))
    high = pivot(legal, 'high')
    low = pivot(legal, 'low')
    amount = pivot(legal, 'amount').where(lambda z: z.gt(0))
    volume = pivot(legal, 'volume').where(lambda z: z.gt(0))
    turn = pivot(legal_f, 'turn')
    ret = close / close.shift(1) - 1
    intraday = close / opening - 1
    overnight = opening / close.shift(1) - 1
    range_ratio = (high - low) / close
    location = (close - low) / (high - low).where(high.ne(low))
    extended = []
    newranks = []
    rawhist = []
    smoothed = []

    def add(name, matrix, group, raw=True):
        # These derived feature ranks are outputs only: later smoothing uses
        # base columns, not this matrix. A single-date request therefore needs
        # masking and cross-sectional ranking only on its requested date.
        if target_date is not None:
            matrix = matrix.loc[[target_date]]
            output_rows = np.zeros(len(ri), dtype=np.intp)
            membership = member.loc[[target_date]]
        else:
            output_rows = ri
            membership = member
        # These frames already have identical date/code axes. Operate on
        # their values to avoid rebuilding several Pandas managers per feature.
        if not matrix.index.equals(membership.index) or not matrix.columns.equals(membership.columns):
            raise ValueError('Feature/member axes mismatch')
        values = matrix.to_numpy(copy=True)
        values[~np.isfinite(values) | ~membership.to_numpy()] = np.nan
        if raw:
            rawname = 'raw_' + name
            raw_values = np.where(np.abs(values) < 1e-12, 0.0, values)
            columns[rawname] = raw_values[output_rows, ci].astype(np.float32)
            rawhist.append(rawname)
        # Preserve average-tie Pandas ranking and original extended precision.
        rank_input = values if group is extended else values.round(10)
        ranked = pd.DataFrame(rank_input).rank(axis=1, pct=True, method='average')
        columns[name] = ranked.to_numpy()[output_rows, ci].astype(np.float32)
        group.append(name)
    # Cache within this call only. Retain the source object to prevent id reuse.
    _statistics = {}

    def _stat(a, w, op):
        key = (id(a), w, op)
        if key not in _statistics:
            _statistics[key] = (a, _fast_rolling(a, w, w, op))
        return _statistics[key][1]

    def mean(a, w):
        return _stat(a, w, 'mean')

    def std(a, w):
        return _stat(a, w, 'std')

    def safe(a):
        return a.where(a.gt(0))

    def finite_corr(a, b, w):
        # Local centered moments have no cumulative state before the window;
        # unlike pandas rolling corr they are bitwise stable under truncation.
        result = np.full(a.shape, np.nan)
        if len(a) < w:
            return pd.DataFrame(result, index=a.index, columns=a.columns)
        av = np.lib.stride_tricks.sliding_window_view(a.to_numpy(), w, axis=0)
        bv = np.lib.stride_tricks.sliding_window_view(b.to_numpy(), w, axis=0)
        output_slice = slice(w - 1, None)
        if target_date is not None:
            position = dates.get_loc(target_date)
            if position < w - 1:
                return pd.DataFrame(result, index=a.index, columns=a.columns)
            av = av[position - w + 1:position - w + 2]
            bv = bv[position - w + 1:position - w + 2]
            output_slice = slice(position, position + 1)
        for start in range(0, a.shape[1], 64):
            aa = av[:, start:start + 64, :]
            bb = bv[:, start:start + 64, :]
            da = aa - aa.mean(axis=2, keepdims=True)
            db = bb - bb.mean(axis=2, keepdims=True)
            den = np.sqrt((da * da).sum(axis=2) * (db * db).sum(axis=2))
            numerator = (da * db).sum(axis=2)
            result[output_slice, start:start + 64] = np.divide(numerator, den, out=np.full_like(den, np.nan), where=den > 0)
        return pd.DataFrame(result, index=a.index, columns=a.columns)
    # Exact definitions/order from detailed_features.py (subject to float32).
    for w in (1, 5, 20, 60):
        add(f'hist_ret_{w}', close / close.shift(w) - 1, extended)
    for w in (5, 20):
        add(f'hist_vol_{w}', std(ret, w), extended)
    add('hist_intraday', intraday, extended)
    add('hist_overnight', overnight, extended)
    add('hist_range_ratio', range_ratio, extended)
    add('hist_close_location', location, extended)
    for w in (5, 20):
        add(f'hist_amount_ratio_{w}', amount / mean(amount, w), extended)
    for w in (5, 20):
        add(f'hist_turn_mean_{w}', mean(turn, w), extended)
    add('hist_turn_ratio_20', turn / safe(mean(turn, 20)), extended)
    add('hist_illiquidity_20', mean(ret.abs() / amount * 100000000.0, 20), extended)
    add('hist_downside_rms_20', mean(ret.clip(upper=0).pow(2), 20).pow(0.5), extended)
    add('hist_ma_gap_20', close / mean(close, 20) - 1, extended)
    add('hist_valid_fraction_60', _fast_rolling(close.notna(), 60, 1, 'sum') / 60, extended)
    add('hist_suspend_fraction_20', mean(suspended, 20), extended)
    # Lagged close/open returns and split intraday/overnight reversals.
    for w in (2, 3, 10, 40, 119):
        add(f'px_ret_{w}', close / close.shift(w) - 1, newranks)
    for w in (1, 2, 5, 20, 60):
        add(f'px_open_ret_{w}', opening / opening.shift(w) - 1, newranks)
    for lag in (1, 2, 3, 4, 5):
        add(f'px_ret_lag_{lag}', ret.shift(lag), newranks)
    for w in (3, 5, 10, 20, 60):
        add(f'px_intraday_mean_{w}', mean(intraday, w), newranks)
        add(f'px_overnight_mean_{w}', mean(overnight, w), newranks)
        add(f'px_range_mean_{w}', mean(range_ratio, w), newranks)
        add(f'px_location_mean_{w}', mean(location, w), newranks)
    add('px_upper_shadow', (high - np.maximum(opening, close)) / close, newranks)
    add('px_lower_shadow', (np.minimum(opening, close) - low) / close, newranks)
    add('px_gap_vs_intraday', overnight - intraday, newranks)
    add('px_intraday_range', (close - opening) / (high - low).where(high.ne(low)), newranks)
    # Longest return distribution is 119 returns, requiring 120 prices.
    for w in (10, 40, 60, 119):
        add(f'px_vol_{w}', std(ret, w), newranks)
    for w in (5, 60):
        add(f'px_downside_rms_{w}', mean(ret.clip(upper=0).pow(2), w).pow(0.5), newranks)
    for w in (20, 60):
        add(f'px_skew_{w}', _fast_rolling(ret, w, w, 'skew'), newranks)
        add(f'px_maxret_{w}', _fast_rolling(ret, w, w, 'max'), newranks)
        add(f'px_minret_{w}', _fast_rolling(ret, w, w, 'min'), newranks)
        add(f'px_positive_fraction_{w}', mean(ret.gt(0).astype(float).where(ret.notna()), w), newranks)
    add('px_vol_ratio_5_20', std(ret, 5) / safe(std(ret, 20)), newranks)
    add('px_vol_ratio_20_60', std(ret, 20) / safe(std(ret, 60)), newranks)
    for w in (5, 10, 40, 60, 120):
        add(f'px_ma_gap_{w}', close / mean(close, w) - 1, newranks)
    for w in (5, 10, 20, 40, 60, 120):
        hi = _fast_rolling(high, w, w, 'max')
        lo = _fast_rolling(low, w, w, 'min')
        add(f'px_range_position_{w}', (close - lo) / (hi - lo).where(hi.ne(lo)), newranks)
        add(f'px_drawdown_{w}', close / hi - 1, newranks)
    for w in (10, 20, 60):
        add(f'px_trend_efficiency_{w}', (close / close.shift(w) - 1) / safe(_fast_rolling(ret.abs(), w, w, 'sum')), newranks)
        add(f'px_short_vs_long_{w}', close / close.shift(w) - 1 - (close / close.shift(3) - 1), newranks)
        add(f'px_risk_scaled_ret_{w}', (close / close.shift(w) - 1) / safe(std(ret, w)), newranks)
    # Turnover and dollar liquidity avoid adjusted-price / raw-volume mixing.
    add('liq_log_amount', np.log(amount), newranks)
    add('liq_log_volume', np.log(volume), newranks)
    for w in (10, 40, 60, 120):
        add(f'liq_amount_ratio_{w}', amount / mean(amount, w), newranks)
    for w in (5, 10, 20, 60, 120):
        add(f'liq_volume_ratio_{w}', volume / mean(volume, w), newranks)
    for w in (3, 10, 60, 120):
        add(f'liq_turn_mean_{w}', mean(turn, w), newranks)
    for w in (5, 60):
        add(f'liq_turn_ratio_{w}', turn / safe(mean(turn, w)), newranks)
        add(f'liq_illiquidity_{w}', mean(ret.abs() / amount * 100000000.0, w), newranks)
    for w in (5, 20, 60):
        add(f'liq_amount_cv_{w}', std(amount, w) / mean(amount, w), newranks)
        add(f'liq_signed_amount_{w}', mean(amount * np.sign(ret), w) / mean(amount, w), newranks)
        add(f'liq_amount_return_corr_{w}', finite_corr(ret, np.log(amount), w), newranks)
    add('liq_turn_change', turn / safe(turn.shift(1)) - 1, newranks)
    add('liq_turn_cv_20', std(turn, 20) / safe(mean(turn, 20)), newranks)
    add('liq_amount_trend_5_20', mean(amount, 5) / mean(amount, 20) - 1, newranks)
    add('liq_amount_trend_20_60', mean(amount, 20) / mean(amount, 60) - 1, newranks)
    add('liq_reversal_amount_interaction', ret * (amount / mean(amount, 20)), newranks)
    add('liq_reversal_turn_interaction', ret * (turn / safe(mean(turn, 20))), newranks)
    add('state_valid_fraction_120', _fast_rolling(close.notna(), 120, 1, 'sum') / 120, newranks)
    add('state_suspend_fraction_60', mean(suspended, 60), newranks)
    # Daily cross-sectional ranks are smoothed only with a finite rolling window.
    # Re-ranking the average on t uses t's actual constituents, not a future pool.
    smooth_sources = ['turn', 'momentum_5', 'reversal_5', 'volatility_5', 'cci_14', 'rsi_12', 'bias_20', 'macd_hist_12_26_9', 'barra_动量', 'barra_残差波动', 'barra_流动性', 'barra_账面市值比']
    for source in smooth_sources:
        z = pivot(base, source)
        for w in (3, 10, 20):
            add(f'smooth_{source}_{w}', mean(z, w), smoothed, raw=False)
        add(f'delta_{source}_5', z - z.shift(5), smoothed, raw=False)
    output = pd.concat([selected.reset_index(drop=True), pd.DataFrame(columns)], axis=1)
    # Useful ablation groups are defined by semantic families, not result selection.
    groups = {'base43': basecols, 'barra10': [c for c in basecols if c.startswith('barra_')], 'extended63': basecols + extended, 'raw_base43': rawcols, 'price_rank': basecols + extended + newranks, 'price_raw': rawcols + rawhist, 'smooth': basecols + extended + smoothed, 'rich': basecols + extended + newranks + smoothed, 'mixed': basecols + extended + newranks + smoothed + rawhist, 'all': list(columns), 'history_only': extended + newranks}
    groups['rich61'] = [c for c in groups['rich'] if FEATURE_LOOKBACK[c] <= 61]
    groups['mixed61'] = [c for c in groups['mixed'] if FEATURE_LOOKBACK[c] <= 61]
    groups['rich21'] = [c for c in groups['rich'] if FEATURE_LOOKBACK[c] <= 21]
    if set(columns) != set(FEATURE_LOOKBACK):
        raise AssertionError('Feature definitions and explicit lookbacks disagree')
    actual = {name: max((FEATURE_LOOKBACK[c] for c in cs)) for name, cs in groups.items()}
    if actual != GROUP_LOOKBACK:
        raise AssertionError(f'Group lookbacks disagree: {actual}')
    return (output, groups)
# Explicit dependency accounting: a w-day return or w observations of one-day
# returns needs w+1 prices. A rolling statistic of a contemporaneous value needs
# w dates. Supplied Factor33 / Barra fields are legal current-day inputs.

def _make_lookbacks():
    factor_names = ['turn', 'change_ratio', 'daily_return', 'momentum_5', 'reversal_5', 'volatility_5', 'sma_20', 'ema_20', 'macd_diff_12_26_9', 'macd_dea_12_26_9', 'macd_hist_12_26_9', 'rsi_12', 'kdj_k_9_3_3', 'kdj_d_9_3_3', 'bias_20', 'cci_14', 'atr_14', 'total_market_cap', 'float_market_cap', 'pe_ttm', 'pb', 'ps_ttm', 'roe_avg_ttm', 'roa_avg_ttm', 'gross_profit_rate_ttm', 'net_profit_rate_ttm', 'debt_to_asset_lf', 'current_ratio_lf', 'beta_000300SH_22', 'list_days', 'is_suspend', 'is_st', 'is_new']
    barra_names = ['市值', '贝塔', '动量', '残差波动', '非线性市值', '账面市值比', '流动性', '盈利', '成长', '杠杆']
    bounds = {c: 1 for c in factor_names + ['barra_' + c for c in barra_names]}
    historical = {'hist_intraday': 1, 'hist_overnight': 2, 'hist_range_ratio': 1, 'hist_close_location': 1, 'hist_turn_ratio_20': 20, 'hist_illiquidity_20': 21, 'hist_downside_rms_20': 21, 'hist_ma_gap_20': 20, 'hist_valid_fraction_60': 60, 'hist_suspend_fraction_20': 20}
    for w in (1, 5, 20, 60):
        historical[f'hist_ret_{w}'] = w + 1
    for w in (5, 20):
        historical[f'hist_vol_{w}'] = w + 1
        historical[f'hist_amount_ratio_{w}'] = w
        historical[f'hist_turn_mean_{w}'] = w
    extra = {'px_upper_shadow': 1, 'px_lower_shadow': 1, 'px_gap_vs_intraday': 2, 'px_intraday_range': 1, 'px_vol_ratio_5_20': 21, 'px_vol_ratio_20_60': 61, 'liq_log_amount': 1, 'liq_log_volume': 1, 'liq_turn_change': 2, 'liq_turn_cv_20': 20, 'liq_amount_trend_5_20': 20, 'liq_amount_trend_20_60': 60, 'liq_reversal_amount_interaction': 20, 'liq_reversal_turn_interaction': 20, 'state_valid_fraction_120': 120, 'state_suspend_fraction_60': 60}
    for w in (2, 3, 10, 40, 119):
        extra[f'px_ret_{w}'] = w + 1
    for w in (1, 2, 5, 20, 60):
        extra[f'px_open_ret_{w}'] = w + 1
    for lag in (1, 2, 3, 4, 5):
        extra[f'px_ret_lag_{lag}'] = lag + 2
    for w in (3, 5, 10, 20, 60):
        for prefix in ('intraday_mean', 'range_mean', 'location_mean'):
            extra[f'px_{prefix}_{w}'] = w
        extra[f'px_overnight_mean_{w}'] = w + 1
    for w in (10, 40, 60, 119):
        extra[f'px_vol_{w}'] = w + 1
    for w in (5, 60):
        extra[f'px_downside_rms_{w}'] = w + 1
    for w in (20, 60):
        for prefix in ('skew', 'maxret', 'minret', 'positive_fraction'):
            extra[f'px_{prefix}_{w}'] = w + 1
    for w in (5, 10, 40, 60, 120):
        extra[f'px_ma_gap_{w}'] = w
    for w in (5, 10, 20, 40, 60, 120):
        extra[f'px_range_position_{w}'] = w
        extra[f'px_drawdown_{w}'] = w
    for w in (10, 20, 60):
        for prefix in ('trend_efficiency', 'short_vs_long', 'risk_scaled_ret'):
            extra[f'px_{prefix}_{w}'] = w + 1
    for w in (10, 40, 60, 120):
        extra[f'liq_amount_ratio_{w}'] = w
    for w in (5, 10, 20, 60, 120):
        extra[f'liq_volume_ratio_{w}'] = w
    for w in (3, 10, 60, 120):
        extra[f'liq_turn_mean_{w}'] = w
    for w in (5, 60):
        extra[f'liq_turn_ratio_{w}'] = w
        extra[f'liq_illiquidity_{w}'] = w + 1
    for w in (5, 20, 60):
        extra[f'liq_amount_cv_{w}'] = w
        extra[f'liq_signed_amount_{w}'] = w + 1
        extra[f'liq_amount_return_corr_{w}'] = w + 1
    smooth = {}
    for source in ['turn', 'momentum_5', 'reversal_5', 'volatility_5', 'cci_14', 'rsi_12', 'bias_20', 'macd_hist_12_26_9', 'barra_动量', 'barra_残差波动', 'barra_流动性', 'barra_账面市值比']:
        for w in (3, 10, 20):
            smooth[f'smooth_{source}_{w}'] = w
        smooth[f'delta_{source}_5'] = 6
    bounds.update(historical)
    bounds.update(extra)
    # Raw copies have exactly the same dependency as their ranked forms.
    bounds.update({'raw_' + c: w for c, w in list(bounds.items())})
    bounds.update(smooth)
    return bounds
FEATURE_LOOKBACK = _make_lookbacks()
GROUP_LOOKBACK = {'base43': 1, 'barra10': 1, 'extended63': 61, 'raw_base43': 1, 'price_rank': 120, 'price_raw': 120, 'smooth': 61, 'rich': 120, 'mixed': 120, 'all': 120, 'history_only': 120, 'rich61': 61, 'mixed61': 61, 'rich21': 21}
