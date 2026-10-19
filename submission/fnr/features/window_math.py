import numpy as np

def stats(x, y, mask, risk_weighted=False):
    a = []
    b = []
    for t in range(len(x)):
        xx = x[t, mask[t]].astype(np.float64)
        yy = y[t, mask[t]]
        n = max(len(xx), 1)
        if risk_weighted:
            weights = np.clip(0.5 - xx[:, 22], 0.1, 1.0)
            n = max(weights.sum(), 1e-12)
            a.append(xx.T @ (xx * weights[:, None]) / n)
            b.append(xx.T @ (yy * weights) / n)
        else:
            a.append(xx.T @ xx / n)
            b.append(xx.T @ yy / n)
    return (np.array(a), np.array(b))

def solve_at(p, t, prior, penalty, statistics, cfg):
    lookback = cfg['lookback']
    start = max(lookback - 1, t - 119 + lookback - 1, t - 1 - cfg['window'])
    ix = np.arange(start, t - 1)
    if len(ix) < 10:
        return prior.copy()
    w = 2.0 ** (-(t - 2 - ix) / cfg['half_life'])
    w /= w.sum()
    a = np.einsum('t,tij->ij', w, statistics[0][ix])
    b = np.einsum('t,ti->i', w, statistics[1][ix])
    return np.linalg.solve(a + penalty, b + penalty @ prior)
