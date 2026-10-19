"""Fixed trailing windows with cached bounds and unchanged Pandas aggregation.

For Pandas 2.2.3, replace ``frame.rolling(w, min_periods=m).mean()`` with
``rolling(frame, w, m, 'mean')``. The Cython aggregation receives the same full
input column and the same start/end arrays. This does not slice prefixes,
recompute reductions in NumPy, or alter the order of floating-point operations.
Only positive integer, trailing, right-closed windows with step=1 are supported.
The bounded cache stores index arrays only, never market data or predictions.
"""
from functools import lru_cache
from numbers import Integral
import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

class CachedTrailingIndexer(BaseIndexer):
    """Public custom-indexer API; one pair of bounds shared by all columns."""

    def __init__(self, num_values, window_size):
        super().__init__(window_size=window_size)
        self.num_values = num_values
        self.end = np.arange(1, num_values + 1, dtype=np.int64)
        self.start = np.maximum(self.end - window_size, 0)

    def get_window_bounds(self, num_values=0, min_periods=None, center=None, closed=None, step=None):
        if num_values != self.num_values or center not in (None, False) or closed not in (None, 'right') or (step not in (None, 1)):
            raise ValueError('cached indexer requires unchanged trailing window shape')
        return (self.start, self.end)

@lru_cache(maxsize=128)
def trailing_indexer(num_values, window_size):
    """Return immutable-in-use bounds; callers must not modify start/end."""
    return CachedTrailingIndexer(num_values, window_size)

def rolling(frame, w, min_periods, op, ddof=1):
    """Apply mean/std/sum/skew/max/min with the ordinary Pandas Cython kernels."""
    if not isinstance(frame, (pd.DataFrame, pd.Series)):
        raise TypeError('frame must be a pandas DataFrame or Series')
    if isinstance(w, bool) or not isinstance(w, Integral) or w < 1:
        raise ValueError('w must be a positive integer')
    if min_periods is None:
        min_periods = w
    if isinstance(min_periods, bool) or not isinstance(min_periods, Integral) or (not 0 <= min_periods <= w):
        raise ValueError('min_periods must be an integer from 0 to w')
    if op not in ('mean', 'std', 'sum', 'skew', 'max', 'min'):
        raise ValueError('unsupported rolling operation')
    obj = frame.rolling(trailing_indexer(len(frame), int(w)), min_periods=int(min_periods))
    if op == 'std':
        return obj.std(ddof=ddof, engine='cython')
    if op == 'skew':
        return obj.skew()
    if op == 'mean':
        return obj.mean(engine='cython')
    if op == 'sum':
        return obj.sum(engine='cython')
    if op == 'max':
        return obj.max(engine='cython')
    return obj.min(engine='cython')
