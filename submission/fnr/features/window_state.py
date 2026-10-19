import numpy as np
import pandas as pd

def state_panel(data, p):
    f = data['MarketData']

    def wide(c):
        return f.pivot(index='日期', columns='代码', values=c).reindex(index=p['days'], columns=p['codes']).where(p['mask'])
    o = wide('open')
    c = wide('close')
    v = wide('amount')
    cc = c / c.shift(1) - 1
    oc = c / o - 1
    gap = o / c.shift(1) - 1
    z = [cc.mean(1), oc.mean(1), gap.mean(1), cc.std(axis=1), oc.std(axis=1), gap.std(axis=1), cc.gt(0).where(p['mask']).mean(1) - 0.5, cc.mean(1).rolling(5, min_periods=5).mean()]
    return np.nan_to_num(np.stack([a.to_numpy(dtype=float) for a in z], -1), nan=0.0, posinf=0.0, neginf=0.0)
