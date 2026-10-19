"""Frozen Gaussian-factor representation plus a bounded relative-stock adapter.

This is a risk-neighbor ranking surrogate, not the organizer's optimizer.
All neighbor selection depends only on contemporaneous allowed features and
frozen model parameters; future labels occur exclusively in adapter loss.
"""
from pathlib import Path
import json
import numpy as np
from . import factor_core as core
CONFIGS = {'C00': dict(neighborhood='none', pair_coefficient=0.0), 'P00': dict(neighborhood='barra', pair_coefficient=0.02), 'P01': dict(neighborhood='factor_barra', pair_coefficient=0.02)}

def config(name):
    return dict(name=name, **CONFIGS[name], width=16, delta_cap=0.05, neighbors=24, epochs=16, learning_rate=0.001, weight_decay=0.001, preserve_coefficient=2.0, pair_temperature=0.1, return_gap_scale=0.02, seed=42, batch_days=16, outer_auxiliary_weight=0.4)

def normalized_distance(x):
    x = np.asarray(x, np.float64)
    q = (x * x).sum(1)
    d = np.maximum(q[:, None] + q[None, :] - 2 * x @ x.T, 0)
    positive = d[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-12]
    return d / (np.median(positive) if len(positive) else 1.0)

def encode(teacher, x, mask):
    """Single-date stock representation and two independently normalized graphs."""
    assert x.shape[0] == 1 and mask.shape[0] == 1
    c = core.components(teacher, x, mask)
    h = np.asarray(c['hidden'][0], np.float32)
    sd = np.sqrt(c['cash_variance'][0])
    u = c['beta'][0] * np.exp(0.5 * c['prior_logvar'][0])[None]
    share = 1 - np.exp(c['idio_logvar'][0]) / c['cash_variance'][0]
    stats = np.column_stack([c['score'][0], c['rank_score'][0], c['alpha'][0] / sd, c['common_mean'][0] / sd, np.log(sd), share, u]).astype(np.float32)
    nodes = np.concatenate([h, stats], 1).astype(np.float32)
    indices = np.flatnonzero(mask[0])
    style = np.asarray(x[0, indices, :10], np.float64)
    style = (style - style.mean(0)) / np.maximum(style.std(0), 1e-06)
    db = normalized_distance(style)
    df = normalized_distance(u[indices])
    graphs = {}
    for name, distance in [('barra', db), ('factor_barra', 0.5 * (db + df))]:
        distance = distance.copy()
        np.fill_diagonal(distance, np.inf)
        k = min(24, max(len(indices) - 1, 0))
        ni = np.zeros((len(mask[0]), 24), np.int16)
        nw = np.zeros(ni.shape, np.float32)
        if k:
            select = np.argsort(distance, axis=1, kind='stable')[:, :k]
            w = np.exp(-np.take_along_axis(distance, select, 1))
            w /= np.maximum(w.sum(1, keepdims=True), 1e-20)
            ni[indices, :k] = indices[select]
            nw[indices, :k] = w.astype(np.float32)
        graphs[name] = (ni, nw)
    nodes[~mask[0]] = 0
    return (nodes, np.asarray(c['score'][0], np.float32), graphs)

def features(nodes, indices, weights, mode):
    if mode == 'none':
        peer = np.zeros_like(nodes)
        relative = np.zeros_like(nodes)
    else:
        if nodes.ndim == 2:
            peer = (nodes[indices] * weights[..., None]).sum(-2)
        else:
            peer = (nodes[np.arange(len(nodes))[:, None, None], indices] * weights[..., None]).sum(-2)
        relative = nodes - peer
    return np.concatenate([nodes, peer, relative], axis=-1).astype(np.float32)

def features_many(nodes, indices, weights, mode):
    # Avoid allocating an entire [days,stocks,neighbors,channels] array.
    out = np.empty((*nodes.shape[:2], 3 * nodes.shape[2]), np.float32)
    for i in range(len(nodes)):
        out[i] = features(nodes[i], indices[i], weights[i], mode)
    return out

def build(cfg, input_dim):
    import torch
    from torch import nn

    class Adapter(nn.Module):

        def __init__(self):
            super().__init__()
            self.config = cfg
            self.first = nn.Linear(input_dim, cfg['width'])
            self.last = nn.Linear(cfg['width'], 1)
            nn.init.zeros_(self.last.weight)
            nn.init.zeros_(self.last.bias)

        def forward(self, x, mask, teacher):
            delta = self.config['delta_cap'] * torch.tanh(self.last(torch.relu(self.first(x))).squeeze(-1))
            delta = delta * mask
            # A cross-sectionally common correction cannot improve the ranking.
            delta = (delta - (delta.sum(1) / mask.sum(1).clamp_min(1))[:, None]) * mask
            return (teacher + delta, delta)
    return Adapter()

def loss(score, delta, b, cfg):
    import torch
    m = b['label_mask'].float()
    den = m.sum(1).clamp_min(1)
    rank = (((score - b['y_rank']) ** 2 * m).sum(1) / den).mean()
    preserve = ((delta.square() * m).sum(1) / den).mean()
    pair = rank.new_zeros(())
    if cfg['pair_coefficient']:
        neighbors = b['neighbors']
        batch = torch.arange(len(score))[:, None, None]
        gap = b['returns'][..., None] - b['returns'][batch, neighbors]
        valid = b['label_mask'][..., None] & b['label_mask'][batch, neighbors]
        w = b['weights'] * valid * torch.clamp(gap.abs() / cfg['return_gap_scale'], max=3)
        logits = (score[..., None] - score[batch, neighbors]) / cfg['pair_temperature']
        raw = torch.nn.functional.softplus(-torch.sign(gap) * logits)
        pair = ((raw * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-12)).mean()
    return (rank + cfg['preserve_coefficient'] * preserve + cfg['pair_coefficient'] * pair, dict(rank=rank, preserve=preserve, pair=pair))

def export(model, dest, feature_pre):
    p = Path(dest)
    p.mkdir(exist_ok=False, parents=True)
    np.savez_compressed(p / 'adapter.npz', **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    (p / 'adapter.json').write_text(json.dumps(dict(config=model.config, preprocessing=feature_pre), indent=2) + '\n')

def load_adapter(path):
    p = Path(path)
    meta = json.loads((p / 'adapter.json').read_text())
    with np.load(p / 'adapter.npz', allow_pickle=False) as f:
        weights = {k: f[k].copy() for k in f.files}
    return (meta, weights)

def apply_adapter(loaded, features, mask, teacher):
    meta, s = loaded
    cfg = meta['config']
    pre = meta['preprocessing']
    x = np.clip(np.nan_to_num((features - np.asarray(pre['mean'], np.float32)) / np.asarray(pre['std'], np.float32), nan=0, posinf=0, neginf=0), -8, 8)
    hidden = np.maximum(x @ s['first.weight'].T + s['first.bias'], 0)
    delta = cfg['delta_cap'] * np.tanh((hidden @ s['last.weight'].T + s['last.bias']).squeeze(-1)) * mask
    delta = (delta - (delta.sum(1) / np.maximum(mask.sum(1), 1))[:, None]) * mask
    return ((teacher + delta).astype(np.float32), delta)

def load_numpy(path):
    p = Path(path)
    return (core.load_numpy(p / 'teacher'), load_adapter(p / 'adapter'))

def predict_numpy(path, x, mask, history=None):
    teacher, adapter = load_numpy(path) if not isinstance(path, tuple) else path
    mode = adapter[0]['config']['neighborhood']
    result = []
    for i in range(len(x)):
        nodes, baseline, graphs = encode(teacher, x[i:i + 1], mask[i:i + 1])
        indices, weights = graphs['barra' if mode == 'none' else mode]
        f = features(nodes, indices, weights, mode)
        score, _ = apply_adapter(adapter, f[None], mask[i:i + 1], baseline[None])
        result.append(score[0])
    return {'score': np.asarray(result, np.float32)}
