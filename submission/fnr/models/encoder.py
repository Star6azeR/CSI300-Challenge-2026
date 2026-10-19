"""Small permutation-equivariant cross-sectional networks and NumPy inference.

These are StockMixer-/MASTER-inspired ablations, not published-model replicas.
The stock mask is an as-of input mask; it MUST NOT be a future-label mask.
Importing this module or using predict_numpy does not import Torch.
"""
from pathlib import Path
import json
import numpy as np
DEFAULTS = dict(family='set_mixer', width=32, heads=4, dropout=0.05, market_gate=False, history_length=20, history_dim=16, temporal_hidden=8, temporal_patches=[1, 5], layer_norm_eps=1e-05)

def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    if cfg['family'] not in ('set_mixer', 'market_attention', 'temporal_mixer'):
        raise ValueError('Unsupported mixer family')
    if cfg['width'] % cfg['heads']:
        raise ValueError('width must be divisible by heads')
    if cfg['family'] == 'temporal_mixer':
        if not 1 <= cfg['history_length'] <= 60:
            raise ValueError('History length must be 1..60 (base lookback <=61)')
        if any((cfg['history_length'] % p for p in cfg['temporal_patches'])):
            raise ValueError('History length must be divisible by temporal patches')
    return cfg

def build(config, input_dim):
    import torch
    from torch import nn
    cfg = normalize_config(config)
    w = cfg['width']

    class CrossSectionModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.config = cfg
            self.input_dim = input_dim
            self.enc = nn.Linear(input_dim, w)
            self.enc_norm = nn.LayerNorm(w, eps=cfg['layer_norm_eps'])
            if cfg['market_gate']:
                self.gate1 = nn.Linear(2 * input_dim, w)
                self.gate2 = nn.Linear(w, input_dim)
                nn.init.zeros_(self.gate2.weight)
                nn.init.zeros_(self.gate2.bias)
            if cfg['family'] == 'market_attention':
                self.qkv = nn.Linear(w, 3 * w)
                self.attn_out = nn.Linear(w, w)
                self.attn_norm = nn.LayerNorm(w, eps=cfg['layer_norm_eps'])
            if cfg['family'] == 'temporal_mixer':
                self.history_enc = nn.Linear(cfg['history_dim'] + 1, w)
                self.time1 = nn.ModuleList([nn.Linear(cfg['history_length'] // p, cfg['temporal_hidden']) for p in cfg['temporal_patches']])
                self.time2 = nn.ModuleList([nn.Linear(cfg['temporal_hidden'], 1) for p in cfg['temporal_patches']])
                self.time_merge = nn.Linear(w * len(cfg['temporal_patches']), w)
                self.time_norm = nn.LayerNorm(w, eps=cfg['layer_norm_eps'])
            self.mix1 = nn.Linear(4 * w, 2 * w)
            self.mix2 = nn.Linear(2 * w, w)
            self.mix_norm = nn.LayerNorm(w, eps=cfg['layer_norm_eps'])
            self.head = nn.Linear(w, 1)
            self.drop = nn.Dropout(cfg['dropout'])

        def pool(self, x, mask):
            weights = mask.to(x.dtype).unsqueeze(-1)
            denom = weights.sum(1, keepdim=True).clamp_min(1)
            mean = (x * weights).sum(1, keepdim=True) / denom
            var = ((x - mean).square() * weights).sum(1, keepdim=True) / denom
            std = (var + 1e-05).sqrt()
            return (mean, std)

        def forward(self, x, mask, history=None):
            if x.ndim != 3 or mask.shape != x.shape[:2]:
                raise ValueError('Expected x[B,N,F] and as-of mask[B,N]')
            mask = mask.bool()
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x * mask.unsqueeze(-1)
            if cfg['market_gate']:
                mu, sd = self.pool(x, mask)
                gate = 2 * torch.sigmoid(self.gate2(torch.relu(self.gate1(torch.cat([mu, sd], -1)))))
                x = x * gate
            h = torch.relu(self.enc_norm(self.enc(x))) * mask.unsqueeze(-1)
            if cfg['family'] == 'market_attention':
                b, n, _ = h.shape
                q, k, v = self.qkv(h).reshape(b, n, 3, cfg['heads'], w // cfg['heads']).unbind(2)
                q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
                logits = q @ k.transpose(-1, -2) / (w // cfg['heads']) ** 0.5
                logits = logits.masked_fill(~mask[:, None, None, :], -1000000000.0)
                weights = torch.softmax(logits, -1) * mask[:, None, None, :]
                z = (weights @ v).transpose(1, 2).reshape(b, n, w)
                h = self.attn_norm(h + self.drop(self.attn_out(z))) * mask.unsqueeze(-1)
            if cfg['family'] == 'temporal_mixer':
                if history is None:
                    raise ValueError('temporal_mixer requires as-of historical features')
                hv = history['values'] if isinstance(history, dict) else history
                hm = history.get('mask', torch.isfinite(hv).any(-1)) if isinstance(history, dict) else torch.isfinite(hv).any(-1)
                if hv.shape != (*x.shape[:2], cfg['history_length'], cfg['history_dim']) or hm.shape != hv.shape[:-1]:
                    raise ValueError('History shape differs from configured [B,N,L,C]')
                hv = torch.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0) * hm.unsqueeze(-1)
                hv = torch.cat([hv, hm.to(hv.dtype).unsqueeze(-1)], -1)
                z = torch.relu(self.history_enc(hv)) * hm.unsqueeze(-1)
                z = z.transpose(-1, -2)
                ts = []
                for p, t1, t2 in zip(cfg['temporal_patches'], self.time1, self.time2):
                    pooled = z.reshape(*z.shape[:-1], cfg['history_length'] // p, p).mean(-1)
                    ts.append(t2(torch.relu(t1(pooled))).squeeze(-1))
                z = self.time_merge(torch.cat(ts, -1))
                h = self.time_norm(h + self.drop(z)) * mask.unsqueeze(-1)
            mu, sd = self.pool(h, mask)
            local = torch.cat([h, mu.expand_as(h), sd.expand_as(h), h - mu], -1)
            z = self.mix2(self.drop(torch.relu(self.mix1(local))))
            h = self.mix_norm(h + self.drop(z))
            return {'score': self.head(h).squeeze(-1) * mask, 'hidden': h}
    return CrossSectionModel()

def export_numpy(model, path, config=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = normalize_config(model.config if config is None else config)
    state = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in model.state_dict().items()}
    np.savez_compressed(path / 'weights.npz', **state)
    meta = dict(format_version=1, architecture='causal_set_and_temporal_mixer', config=cfg, input_dim=model.input_dim, parameter_count=int(sum((a.size for a in state.values()))), mask_policy='asof feature membership only; never future label validity', numpy_runtime='modules.py', dtype='float32')
    (path / 'model.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return meta

def _linear(x, s, key):
    return x @ s[key + '.weight'].T + s[key + '.bias']

def _norm(x, s, key, eps):
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * s[key + '.weight'] + s[key + '.bias']

def _pool(x, mask):
    weights = mask.astype(np.float32)[..., None]
    denom = np.maximum(weights.sum(1, keepdims=True), 1)
    mean = (x * weights).sum(1, keepdims=True) / denom
    var = ((x - mean) ** 2 * weights).sum(1, keepdims=True) / denom
    return (mean, np.sqrt(var + 1e-05))

def load_numpy(path):
    path = Path(path)
    meta = json.loads((path / 'model.json').read_text())
    with np.load(path / 'weights.npz', allow_pickle=False) as f:
        state = {name: f[name] for name in f.files}
    return (meta, state)

def predict_numpy(path, x, mask, history=None):
    """Pure NumPy inference; path may also be the tuple returned by load_numpy."""
    meta, s = load_numpy(path) if not isinstance(path, tuple) else path
    cfg = meta['config']
    w = cfg['width']
    eps = cfg['layer_norm_eps']
    x = np.asarray(x, np.float32)
    mask = np.asarray(mask, bool)
    if x.ndim != 3 or mask.shape != x.shape[:2] or x.shape[-1] != meta['input_dim']:
        raise ValueError('Expected x[B,N,F] and as-of mask[B,N]')
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * mask[..., None]
    if cfg['market_gate']:
        mu, sd = _pool(x, mask)
        z = _linear(np.maximum(_linear(np.concatenate([mu, sd], -1), s, 'gate1'), 0), s, 'gate2')
        gate = 2 / (1 + np.exp(-np.clip(z, -80, 80)))
        x = x * gate
    h = np.maximum(_norm(_linear(x, s, 'enc'), s, 'enc_norm', eps), 0) * mask[..., None]
    if cfg['family'] == 'market_attention':
        b, n, _ = h.shape
        heads = cfg['heads']
        d = w // heads
        qkv = _linear(h, s, 'qkv').reshape(b, n, 3, heads, d)
        q, k, v = [qkv[:, :, j].transpose(0, 2, 1, 3) for j in range(3)]
        logits = q @ k.transpose(0, 1, 3, 2) / np.float32(d ** 0.5)
        logits = np.where(mask[:, None, None, :], logits, -1000000000.0)
        logits = np.exp(logits - logits.max(-1, keepdims=True))
        weights = logits / logits.sum(-1, keepdims=True) * mask[:, None, None, :]
        z = (weights @ v).transpose(0, 2, 1, 3).reshape(b, n, w)
        h = _norm(h + _linear(z, s, 'attn_out'), s, 'attn_norm', eps) * mask[..., None]
    if cfg['family'] == 'temporal_mixer':
        if history is None:
            raise ValueError('temporal_mixer requires as-of historical features')
        hv = np.asarray(history['values'] if isinstance(history, dict) else history, np.float32)
        hm = np.asarray(history.get('mask', np.isfinite(hv).any(-1)) if isinstance(history, dict) else np.isfinite(hv).any(-1), bool)
        if hv.shape != (*x.shape[:2], cfg['history_length'], cfg['history_dim']) or hm.shape != hv.shape[:-1]:
            raise ValueError('History shape differs from configured [B,N,L,C]')
        hv = np.nan_to_num(hv, nan=0.0, posinf=0.0, neginf=0.0) * hm[..., None]
        hv = np.concatenate([hv, hm.astype(np.float32)[..., None]], -1)
        z = np.maximum(_linear(hv, s, 'history_enc'), 0) * hm[..., None]
        z = z.swapaxes(-1, -2)
        ts = []
        for i, p in enumerate(cfg['temporal_patches']):
            pooled = z.reshape(*z.shape[:-1], cfg['history_length'] // p, p).mean(-1)
            z1 = np.maximum(_linear(pooled, s, 'time1.' + str(i)), 0)
            ts.append(_linear(z1, s, 'time2.' + str(i)).squeeze(-1))
        z = _linear(np.concatenate(ts, -1), s, 'time_merge')
        h = _norm(h + z, s, 'time_norm', eps) * mask[..., None]
    mu, sd = _pool(h, mask)
    local = np.concatenate([h, np.broadcast_to(mu, h.shape), np.broadcast_to(sd, h.shape), h - mu], -1)
    z = _linear(np.maximum(_linear(local, s, 'mix1'), 0), s, 'mix2')
    h = _norm(h + z, s, 'mix_norm', eps)
    return {'score': _linear(h, s, 'head').squeeze(-1) * mask, 'hidden': h}
INITIAL_CONFIGS = [dict(family='set_mixer', width=32, heads=4, market_gate=False, dropout=0.05), dict(family='market_attention', width=32, heads=4, market_gate=True, dropout=0.05), dict(family='temporal_mixer', width=32, heads=4, market_gate=False, dropout=0.05, history_length=20, history_dim=16, temporal_hidden=8, temporal_patches=[1, 5])]
