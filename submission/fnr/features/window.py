"""Per-call ridge inference under the organizer clarification. No persistent state."""
import numpy as np
import pandas as pd
BAR = ['市值', '贝塔', '动量', '残差波动', '非线性市值', '账面市值比', '流动性', '盈利', '成长', '杠杆']
FUND = ['pe_ttm', 'pb', 'roe_avg_ttm', 'roa_avg_ttm', 'debt_to_asset_lf', 'float_market_cap']
TECH = ['ret1', 'ret5', 'ret20', 'intraday', 'overnight', 'vol5', 'vol20', 'amount_ratio20', 'turn_ratio20']
NAMES = ['barra_' + x for x in BAR] + FUND + TECH
LOOKBACK = 21

def panels(data, as_of_date=None):
    if as_of_date is not None:
        t = int(as_of_date.replace('-', ''))
        dates = np.sort(data['MarketData'].loc[lambda f: f['日期'].le(t), '日期'].unique())[-120:]
        data = {k: f.loc[f['日期'].isin(dates)].copy() for k, f in data.items()}
        if not len(dates) or dates[-1] != t:
            raise ValueError('Missing current date')
    else:
        dates = np.sort(data['MarketData']['日期'].unique())
    for k in ['MarketData', 'Barra_factor', 'Factor33_winsor']:
        if data[k].duplicated(['日期', '代码']).any():
            raise ValueError('Duplicate input keys')
    codes = np.array(sorted(data['Barra_factor']['代码'].unique()))

    def wide(key, col):
        return data[key].pivot(index='日期', columns='代码', values=col).reindex(index=dates, columns=codes)
    membership = wide('Barra_factor', '市值').copy()
    rows = data['Barra_factor'][['日期', '代码']].assign(_member=1)
    membership = rows.pivot(index='日期', columns='代码', values='_member').reindex(index=dates, columns=codes).eq(1)
    suspended = wide('MarketData', 'is_suspend')
    opening = wide('MarketData', 'open')
    opening = opening.where(opening.gt(0) & suspended.eq(0))
    close = wide('MarketData', 'close')
    close = close.where(close.gt(0) & suspended.eq(0))
    ret = close / close.shift(1) - 1
    amount = wide('MarketData', 'amount').where(lambda f: f.gt(0) & suspended.eq(0))
    turn = wide('Factor33_winsor', 'turn').where(lambda f: f.gt(0) & suspended.eq(0))
    raw = [wide('Barra_factor', c) for c in BAR] + [wide('Factor33_winsor', c) for c in FUND]
    raw += [ret, close / close.shift(5) - 1, close / close.shift(20) - 1, close / opening - 1, opening / close.shift(1) - 1, ret.rolling(5, min_periods=5).std(ddof=0), ret.rolling(20, min_periods=20).std(ddof=0), amount / amount.rolling(20, min_periods=20).mean() - 1, turn / turn.rolling(20, min_periods=20).mean() - 1]
    x = []
    for f in raw:
        # Full historical as-of cross-section, independent of label availability.
        f = f.replace([np.inf, -np.inf], np.nan).where(membership)
        z = f.rank(axis=1, pct=True)
        z = z.sub(z.mean(axis=1), axis=0)
        x.append(z.fillna(0).to_numpy(np.float32))
    x = np.stack(x, -1)
    mask = membership.to_numpy(bool)
    realized = opening.shift(-2) / opening.shift(-1) - 1
    valid = membership & np.isfinite(realized)
    y = realized.where(valid).rank(axis=1, pct=True)
    y = y.sub(y.mean(axis=1), axis=0).fillna(0).to_numpy(np.float32)
    return dict(days=np.asarray(dates), codes=codes, x=x, mask=mask, y=y, raw_returns=realized.where(valid).fillna(0).to_numpy(np.float32), label_mask=valid.to_numpy(bool))
