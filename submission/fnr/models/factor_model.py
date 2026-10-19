"""Low-rank conditional Gaussian return model with an anchored ranking task.

Inspired by probabilistic stock factor models; analytic Gaussian marginalization
replaces a learned variational posterior. This is not a FactorVAE replication.
Forward uses historical features only; future returns occur exclusively in loss.
"""
from pathlib import Path
import json
import numpy as np

def encoder():
    from . import encoder as module
    return module

def build(config, input_dim):
    import torch
    from torch import nn
    enc = encoder()

    class FactorNet(nn.Module):

        def __init__(self):
            super().__init__()
            self.config = dict(config)
            self.input_dim = input_dim
            self.base = enc.build(config, input_dim)
            # Matched zero-auxiliary control preserves the base dropout RNG stream.
            rng = torch.get_rng_state()
            width = config.get('width', 32)
            k = config.get('factors', 3)
            self.alpha = nn.Linear(width, 1)
            self.beta = nn.Linear(width, k)
            self.idio = nn.Linear(width, 1)
            self.prior = nn.Linear(2 * width, 2 * k)
            torch.set_rng_state(rng)

        def forward(self, x, mask, history=None):
            out = self.base(x, mask, history)
            h = out['hidden']
            valid = mask[..., None]
            n = valid.sum(1).clamp_min(1)
            mean = (h * valid).sum(1) / n
            std = (((h - mean[:, None]) ** 2 * valid).sum(1) / n + 1e-05).sqrt()
            prior = self.prior(torch.cat([mean, std], -1))
            mu, logvar = prior.chunk(2, -1)
            beta = 0.1 * self.beta(h)
            alpha = 0.1 * self.alpha(h).squeeze(-1)
            idio = self.idio(h).squeeze(-1).clamp(-4, 2)
            cash_mean = alpha + (beta * mu[:, None]).sum(-1)
            variance = idio.exp() + (beta.square() * logvar.clamp(-4, 2).exp()[:, None]).sum(-1)
            score = out['score'] + config.get('cash_inference_weight', 0.0) * torch.tanh(cash_mean / variance.sqrt().clamp_min(0.05))
            return dict(score=score * mask, beta=beta, alpha=alpha, prior_mean=mu, prior_logvar=logvar.clamp(-4, 2), idio_logvar=idio, cash_mean=cash_mean, cash_variance=variance)
    return FactorNet()

def training_loss(output, y_rank, y_cash, label_mask, config, context=None):
    import torch
    m = label_mask.to(y_rank.dtype)
    n = m.sum(1).clamp_min(1)
    rank = (((output['score'] - y_rank) ** 2 * m).sum(1) / n).mean()
    if not config.get('factor_weight', 0):
        return rank
    if context is None:
        raise ValueError('Raw legal H1 returns required')
    target = (context['raw_return'] / config.get('return_scale', 0.02)).clamp(-3, 3)
    residual = (target - output['cash_mean']) * m
    diagonal = output['idio_logvar'].exp()
    inverse = m / diagonal
    u = output['beta'] * torch.exp(0.5 * output['prior_logvar'])[:, None]
    k = u.shape[-1]
    small = torch.eye(k, device=u.device)[None] + u.transpose(1, 2) @ (u * inverse[..., None])
    projected = (u * inverse[..., None]).transpose(1, 2) @ residual[..., None]
    chol = torch.linalg.cholesky(small)
    correction = (projected * torch.cholesky_solve(projected, chol)).sum((1, 2))
    quadratic = (residual.square() * inverse).sum(1) - correction
    logdet = (output['idio_logvar'] * m).sum(1) + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(1)
    likelihood = (0.5 * (logdet + quadratic) / n).mean()
    loss = rank + config['factor_weight'] * likelihood
    output['loss_diagnostics'] = {'rank_mse': float(rank.detach()), 'factor_nll': float(likelihood.detach()), 'weighted_factor_nll': float((config['factor_weight'] * likelihood).detach())}
    return loss

def export_numpy(model, path, config=None):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    encoder().export_numpy(model.base, path / 'base', model.config)
    np.savez_compressed(path / 'factors.npz', **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if not k.startswith('base.')})
    (path / 'model.json').write_text(json.dumps({'config': model.config, 'input_dim': model.input_dim}, indent=2) + '\n')

def predict_numpy(path, x, mask, history=None):
    path = Path(path)
    config = json.loads((path / 'model.json').read_text())['config']
    out = encoder().predict_numpy(path / 'base', x, mask, history)
    if not config.get('cash_inference_weight', 0):
        return {'score': out['score']}
    with np.load(path / 'factors.npz', allow_pickle=False) as f:
        state = dict(f)
    h = out['hidden']
    m = mask[..., None]
    n = np.maximum(m.sum(1), 1)
    mean = (h * m).sum(1) / n
    std = np.sqrt(((h - mean[:, None]) ** 2 * m).sum(1) / n + 1e-05)
    linear = lambda v, name: v @ state[name + '.weight'].T + state[name + '.bias']
    mu, logvar = np.split(linear(np.concatenate([mean, std], -1), 'prior'), 2, -1)
    beta = 0.1 * linear(h, 'beta')
    alpha = 0.1 * linear(h, 'alpha').squeeze(-1)
    idio = np.clip(linear(h, 'idio').squeeze(-1), -4, 2)
    cash = alpha + (beta * mu[:, None]).sum(-1)
    variance = np.exp(idio) + (beta ** 2 * np.exp(np.clip(logvar, -4, 2))[:, None]).sum(-1)
    return {'score': (out['score'] + config['cash_inference_weight'] * np.tanh(cash / np.maximum(np.sqrt(variance), 0.05))) * mask}
