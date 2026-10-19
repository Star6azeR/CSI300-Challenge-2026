"""Deterministic, bounded as-of features shared by training and replay.

The longest feature uses t-60..t inclusive (61 exchange dates).
Historical market rows are masked by that date's Barra membership.
"""
import numpy as np
import pandas as pd
from audit_and_pilot import make_features

KEY=['日期','代码']
LOOKBACK=61

def build_features(factors,barra,market,calendar=None,extended=True):
    base,cols=make_features(factors,barra,market)
    if not extended:
        return base,cols
    calendar=sorted(market['日期'].unique()) if calendar is None else sorted(calendar)
    legal=barra[KEY].merge(market,on=KEY,how='left',validate='one_to_one')
    turn=barra[KEY].merge(factors[KEY+['turn']],on=KEY,how='left',validate='one_to_one')
    def pivot(frame,col):
        return frame.pivot(index='日期',columns='代码',values=col).reindex(calendar)
    close=pivot(legal,'close').where(pivot(legal,'is_suspend').eq(0))
    opening=pivot(legal,'open').where(pivot(legal,'is_suspend').eq(0))
    high=pivot(legal,'high');low=pivot(legal,'low')
    amount=pivot(legal,'amount').where(lambda z:z>0)
    turnover=pivot(turn,'turn')
    r=close/close.shift(1)-1
    values={}
    for w in (1,5,20,60):
        values[f'ret_{w}']=close/close.shift(w)-1
    for w in (5,20):
        values[f'vol_{w}']=r.rolling(w,min_periods=w).std(ddof=1)
    values['intraday']=close/opening-1
    values['overnight']=opening/close.shift(1)-1
    values['range_ratio']=(high-low)/close
    values['close_location']=(close-low)/(high-low).where(high.ne(low))
    for w in (5,20):
        values[f'amount_ratio_{w}']=amount/amount.rolling(w,min_periods=w).mean()
    for w in (5,20):
        values[f'turn_mean_{w}']=turnover.rolling(w,min_periods=w).mean()
    values['turn_ratio_20']=turnover/turnover.rolling(20,min_periods=20).mean().where(lambda z:z>0)
    values['illiquidity_20']=(r.abs()/amount*1e8).rolling(20,min_periods=20).mean()
    values['downside_rms_20']=r.clip(upper=0).pow(2).rolling(20,min_periods=20).mean().pow(.5)
    values['ma_gap_20']=close/close.rolling(20,min_periods=20).mean()-1
    values['valid_fraction_60']=close.notna().rolling(60,min_periods=1).sum()/60
    values['suspend_fraction_20']=pivot(legal,'is_suspend').rolling(20,min_periods=20).mean()
    index=pd.MultiIndex.from_frame(base[KEY])
    extras=[]
    for name,matrix in values.items():
        col='hist_'+name
        matrix.columns.name='代码';matrix.index.name='日期'
        series=matrix.stack(future_stack=True)
        base[col]=series.reindex(index).to_numpy()
        base[col]=base[col].replace([np.inf,-np.inf],np.nan)
        base[col]=base.groupby('日期',sort=False)[col].rank(pct=True,method='average')
        extras.append(col)
    return base,cols+extras

def build_labels(market,barra):
    dates=sorted(market['日期'].unique())
    next_day=dict(zip(dates[:-1],dates[1:]));next_two=dict(zip(dates[:-2],dates[2:]))
    y=barra[KEY].copy();y['entry_date']=y['日期'].map(next_day);y['exit_date']=y['日期'].map(next_two)
    for endpoint in ('entry','exit'):
        source=market[KEY+['open','is_suspend']].rename(columns={'日期':endpoint+'_date','open':endpoint+'_open','is_suspend':endpoint+'_suspend'})
        y=y.merge(source,on=[endpoint+'_date','代码'],how='left',validate='many_to_one')
    valid=y.entry_suspend.eq(0)&y.exit_suspend.eq(0)&y.entry_open.gt(0)&y.exit_open.gt(0)&np.isfinite(y.entry_open)&np.isfinite(y.exit_open)
    y['future_return']=(y.exit_open/y.entry_open-1).where(valid)
    y['target']=y.groupby('日期').future_return.rank(pct=True,method='average')
    return y
