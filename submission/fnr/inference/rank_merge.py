"""Preserve RH50's first60 names; rank all other stocks by K3."""
import numpy as np
import pandas as pd

def merge_scores(rh50, k3, head_count=60):
    merged = rh50[['date', 'code', 'score']].merge(k3[['date', 'code', 'score']], on=['date', 'code'], how='outer', validate='one_to_one', suffixes=('_rh50', '_k3'))
    if merged[['score_rh50', 'score_k3']].isna().any().any():
        raise ValueError('Universe mismatch')
    parts = []
    for day, d in merged.groupby('date', sort=True):
        d = d.sort_values(['score_rh50', 'code'], ascending=[False, True]).copy()
        n = len(d)
        k = min(head_count, n)
        head = d.iloc[:k].copy()
        tail = d.iloc[k:].copy()
        head['score'] = (n - k + head.score_rh50.rank(method='average')) / n
        tail['score'] = tail.score_k3.rank(method='average') / n
        result = pd.concat([head, tail]).sort_values('code')
        expected = head.sort_values(['score_rh50', 'code'], ascending=[False, True]).code.tolist()
        actual = result.sort_values(['score', 'code'], ascending=[False, True]).code.iloc[:k].tolist()
        assert expected == actual
        parts.append(result[['date', 'code', 'score']])
    return pd.concat(parts, ignore_index=True)
