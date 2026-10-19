"""Frozen rank representation with rebuilt Gaussian factor use.

R00 adapts factor heads only. R01 additionally learns bounded semantic-group
gates for the factor path, keeping the ranking path fixed. No temporal state.
"""
from pathlib import Path
import json
import numpy as np
from . import encoder

def build(config, input_dim):
    import torch
    from torch import nn

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.config = dict(config)
            self.input_dim = input_dim
            self.base = encoder.build(config, input_dim)
            for p in self.base.parameters():
                p.requires_grad_(config.get('rank_finetune', False))
            w = config['width']
            k = config['factors']
            self.alpha = nn.Linear(w, 1)
            self.beta = nn.Linear(w, k)
            self.idio = nn.Linear(w, 1)
            self.prior = nn.Linear(2 * w, 2 * k)
            self.group_logits = nn.Parameter(torch.zeros(6), requires_grad=config['routing'])
            self.register_buffer('group_index', torch.tensor(config['group_index'], dtype=torch.long))

        def forward(self, x, mask, history=None):
            self.base.eval()
            original = self.base(x, mask)
            if self.config['routing']:
                scales = 1 + 0.2 * torch.tanh(self.group_logits[self.group_index])
                scales = scales.clone()
                scales[self.config['st_index']] = 1.0
                h = self.base(x * scales, mask)['hidden']
            else:
                h = original['hidden']
            valid = mask[..., None]
            n = valid.sum(1).clamp_min(1)
            mean = (h * valid).sum(1) / n
            sd = (((h - mean[:, None]) ** 2 * valid).sum(1) / n + 1e-05).sqrt()
            mu, lv = self.prior(torch.cat([mean, sd], -1)).chunk(2, -1)
            lv = lv.clamp(-4, 2)
            beta = 0.1 * self.beta(h)
            alpha = 0.1 * self.alpha(h).squeeze(-1)
            idio = self.idio(h).squeeze(-1).clamp(-4, 2)
            common = (beta * mu[:, None]).sum(-1)
            cash = alpha + common
            variance = idio.exp() + (beta.square() * lv.exp()[:, None]).sum(-1)
            score = original['score'] + 0.1 * torch.tanh(cash / variance.sqrt().clamp_min(0.05))
            return dict(score=score * mask, rank_score=original['score'], alpha=alpha, beta=beta, prior_mean=mu, prior_logvar=lv, idio_logvar=idio, cash_mean=cash, cash_variance=variance, common_mean=common)
    return Model()

def load_teacher(model, path):
    import torch
    path = Path(path)
    with np.load(path / 'base/weights.npz', allow_pickle=False) as f:
        model.base.load_state_dict({k: torch.from_numpy(f[k].copy()) for k in f.files})
    with np.load(path / 'factors.npz', allow_pickle=False) as f:
        for name, layer in [('alpha', model.alpha), ('beta', model.beta), ('idio', model.idio), ('prior', model.prior)]:
            layer.load_state_dict({s: torch.from_numpy(f[name + '.' + s].copy()) for s in ['weight', 'bias']})

def loss(output, batch, teacher, model, initial, config):
    import torch
    m = batch['label_mask'].to(batch['y_rank'].dtype)
    n = m.sum(1).clamp_min(1)
    rank = (((output['score'] - batch['y_rank']) ** 2 * m).sum(1) / n).mean()
    target = (batch['raw_return'] / 0.02).clamp(-3, 3)
    residual = (target - output['cash_mean']) * m
    inverse = m / output['idio_logvar'].exp()
    u = output['beta'] * torch.exp(0.5 * output['prior_logvar'])[:, None]
    small = torch.eye(config['factors'])[None] + u.transpose(1, 2) @ (u * inverse[..., None])
    projected = (u * inverse[..., None]).transpose(1, 2) @ residual[..., None]
    chol = torch.linalg.cholesky(small)
    quadratic = (residual.square() * inverse).sum(1) - (projected * torch.cholesky_solve(projected, chol)).sum((1, 2))
    logdet = (output['idio_logvar'] * m).sum(1) + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(1)
    nll = (0.5 * (logdet + quadratic) / n).mean()
    preserve = (((output['score'] - teacher) ** 2 * m).sum(1) / n).mean()
    drift = torch.stack([(p - initial[name]).square().mean() for name, p in model.named_parameters() if p.requires_grad and name != 'group_logits']).mean()
    regularize = model.group_logits.square().mean()
    return (rank + 0.05 * nll + 2 * preserve + 0.001 * drift + 0.001 * regularize, {'rank': rank, 'nll': nll, 'preserve': preserve})

def export_numpy(model, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    encoder.export_numpy(model.base, path / 'base', model.config)
    np.savez_compressed(path / 'factors.npz', **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if not k.startswith('base.')})
    (path / 'model.json').write_text(json.dumps({'config': model.config, 'input_dim': model.input_dim}, indent=2) + '\n')

def load_numpy(path):
    p = Path(path)
    config = json.loads((p / 'model.json').read_text())['config']
    with np.load(p / 'factors.npz', allow_pickle=False) as f:
        state = {k: f[k].copy() for k in f.files}
    return (config, encoder.load_numpy(p / 'base'), state)

def components(loaded, x, mask):
    cfg, base, s = loaded
    original = encoder.predict_numpy(base, x, mask)
    if cfg.get('routing', False):
        scales = 1 + 0.2 * np.tanh(s['group_logits'][s['group_index']])
        scales[cfg['st_index']] = 1.0
        h = encoder.predict_numpy(base, x * scales, mask)['hidden']
    else:
        h = original['hidden']
    valid = mask[..., None]
    n = np.maximum(valid.sum(1), 1)
    mean = (h * valid).sum(1) / n
    sd = np.sqrt(((h - mean[:, None]) ** 2 * valid).sum(1) / n + 1e-05)
    linear = lambda v, k: v @ s[k + '.weight'].T + s[k + '.bias']
    mu, lv = np.split(linear(np.concatenate([mean, sd], -1), 'prior'), 2, -1)
    lv = np.clip(lv, -4, 2)
    beta = 0.1 * linear(h, 'beta')
    alpha = 0.1 * linear(h, 'alpha').squeeze(-1)
    idio = np.clip(linear(h, 'idio').squeeze(-1), -4, 2)
    common = (beta * mu[:, None]).sum(-1)
    cash = alpha + common
    v = np.exp(idio) + (beta ** 2 * np.exp(lv)[:, None]).sum(-1)
    score = original['score'] + 0.1 * np.tanh(cash / np.maximum(np.sqrt(v), 0.05))
    return dict(score=score * mask, hidden=original['hidden'], rank_score=original['score'], alpha=alpha, common_mean=common, cash_mean=cash, cash_variance=v, beta=beta, prior_logvar=lv, idio_logvar=idio)

def predict_numpy(path, x, mask, history=None):
    return components(load_numpy(path) if not isinstance(path, tuple) else path, x, mask)

def correlation_precision_score(score, components, strength=0.1):
    """Woodbury inverse of (1-strength)I+strength*predicted correlation.

    Uses the full low-rank covariance, not marginal-risk division alone. This
    changes score ordering, never the organizer's actual portfolio constraints.
    """
    v = np.asarray(components['cash_variance'], float).reshape(-1)
    d = np.exp(np.asarray(components['idio_logvar'], float)).reshape(-1) / v
    beta = np.asarray(components['beta'], float).reshape(len(v), -1)
    lv = np.asarray(components['prior_logvar'], float).reshape(-1)
    u = beta * np.exp(0.5 * lv)[None, :] / np.sqrt(v)[:, None] * np.sqrt(strength)
    diag = 1 - strength + strength * d
    z = np.asarray(score, float) - np.mean(score)
    inv = 1 / diag
    small = np.eye(u.shape[1]) + u.T @ (inv[:, None] * u)
    return inv * z - inv * (u @ np.linalg.solve(small, u.T @ (inv * z)))
