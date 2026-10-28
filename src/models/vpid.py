# coding: utf-8
"""
Predictive PID (V-PID): Partial Information Decomposition of the
collaborative signal Y via V-usable information, learned with a
JEPA-style latent predictor.

Notation
--------
x_v, x_t : frozen visual / text item features
Y        : ID-only collaborative item embedding (pretrain_cf.py),
           standardized per dimension, stop-grad (it is a target).
V        : predictor family = MLP decoders with a diagonal Gaussian
           output. For this family
               H_V(Y | X) = sum_d 0.5 * log(2*pi*e * MSE_d(X))
               I_V(X -> Y) = sum_d 0.5 * log(Var_d(Y) / MSE_d(X))
           so every MSE loss below is an empirical predictive
           V-entropy, and the JEPA predictor IS the V-information
           estimator.

Decomposition (all quantities about Y):
    I_v  = I_V(v -> Y)          from  Y_v  = g_r(r_v) + g_uv(u_v)
    I_t  = I_V(t -> Y)          from  Y_t  = g_r(r_t) + g_ut(u_t)
    I_vt = I_V((v,t) -> Y)      from  Y_vt = g_r(r) + g_uv(u_v) + g_ut(u_t) + g_s(s)
    R    = V-redundant information (Kleinman et al., 2021): both
           modalities decode Y through the SHARED decoder g_r and are
           forced to agree, g_r(r_v) ~= g_r(r_t).
    U_v  = I_v - R,   U_t = I_t - R
    S    = I_vt - I_v - I_t + R       (can be negative: "net synergy")
    1 - R2_vt = share of Y no modality explains (collaborative-only).

Exclusivity (adversarial, via gradient reversal):
    g_uv(u_v) must be V-unpredictable from x_t
    g_ut(u_t) must be V-unpredictable from x_v
    g_s(s)    must be V-unpredictable from x_v alone and from x_t alone
Adversaries minimize a scale-free error  MSE / Var(target)  (= 1 - R^2);
the reversed gradient makes the component maximize it, i.e. push the
adversary down to the "predict the mean" baseline = zero V-information.

Sanity check on synthetic data with known R/U/S (run from src/):
    python -m models.vpid
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jepa import MLP, DecompositionHead, SynergyHead, grad_reverse


# ============================================================
# Configuration
# ============================================================

@dataclass
class VPIDConfig:
    visual_input_dim: int = 384
    text_input_dim: int = 384
    y_dim: int = 64

    encoder_hidden_dim: int = 128
    latent_dim: int = 64
    comp_dim: int = 64
    decoder_hidden_dim: int = 64
    adversary_hidden_dim: int = 128
    input_dropout: float = 0.0

    # loss weights
    lambda_single: float = 1.0     # H_V(Y|v) + H_V(Y|t)
    lambda_joint: float = 1.0      # H_V(Y|v,t)
    lambda_r: float = 1.0          # V-redundancy (RINE)
    beta_r: float = 1.0            # agreement weight (after warm-up)
    beta_warmup_steps: int = 500   # RINE: increase beta gradually
    lambda_excl: float = 0.1       # adversarial exclusivity
    grl_lambda: float = 1.0
    eps: float = 1e-3


# ============================================================
# Weighted statistics helpers
# ============================================================

def _wmean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the item axis (dim 0) with optional item weights. x: [N, ...]."""
    if w is None:
        return x.mean(dim=0)
    w = w.view(-1, *([1] * (x.dim() - 1)))
    return (x * w).sum(dim=0) / w.sum().clamp_min(1e-8)


def _per_dim_mse(pred, target, w):
    return _wmean((pred - target).pow(2), w)          # [D]


def _per_dim_var(x, w):
    mu = _wmean(x, w)
    return _wmean((x - mu).pow(2), w)                  # [D]


# ============================================================
# Model
# ============================================================

class PredictivePID(nn.Module):
    def __init__(self, cfg: VPIDConfig = None):
        super().__init__()
        self.cfg = cfg or VPIDConfig()
        c = self.cfg

        # modality encoders (separate weights)
        self.enc_v = MLP(c.visual_input_dim, c.encoder_hidden_dim, c.latent_dim, num_layers=2)
        self.enc_t = MLP(c.text_input_dim, c.encoder_hidden_dim, c.latent_dim, num_layers=2)
        self.in_drop = nn.Dropout(c.input_dropout) if c.input_dropout > 0 else nn.Identity()

        # PID heads
        self.decomp_v = DecompositionHead(c.latent_dim, c.encoder_hidden_dim, c.comp_dim)
        self.decomp_t = DecompositionHead(c.latent_dim, c.encoder_hidden_dim, c.comp_dim)
        self.synergy = SynergyHead(c.latent_dim, c.encoder_hidden_dim, c.latent_dim, c.comp_dim)

        # decoders to Y (g_r is SHARED by both modalities)
        self.g_r = MLP(c.comp_dim, c.decoder_hidden_dim, c.y_dim, num_layers=2)
        self.g_uv = MLP(c.comp_dim, c.decoder_hidden_dim, c.y_dim, num_layers=2)
        self.g_ut = MLP(c.comp_dim, c.decoder_hidden_dim, c.y_dim, num_layers=2)
        self.g_s = MLP(c.comp_dim, c.decoder_hidden_dim, c.y_dim, num_layers=2)

        # adversaries on RAW single-modality inputs
        self.adv_t_uv = MLP(c.text_input_dim, c.adversary_hidden_dim, c.y_dim, num_layers=2)
        self.adv_v_ut = MLP(c.visual_input_dim, c.adversary_hidden_dim, c.y_dim, num_layers=2)
        self.adv_v_s = MLP(c.visual_input_dim, c.adversary_hidden_dim, c.y_dim, num_layers=2)
        self.adv_t_s = MLP(c.text_input_dim, c.adversary_hidden_dim, c.y_dim, num_layers=2)

        self.register_buffer('step', torch.zeros((), dtype=torch.long))

    # --------------------------------------------------------
    def forward(self, x_v: torch.Tensor, x_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_v = self.enc_v(self.in_drop(x_v))
        z_t = self.enc_t(self.in_drop(x_t))
        r_v, u_v = self.decomp_v(z_v)
        r_t, u_t = self.decomp_t(z_t)
        _, s = self.synergy(z_v, z_t)
        r = 0.5 * (r_v + r_t)

        c_rv, c_rt = self.g_r(r_v), self.g_r(r_t)
        c_r = 0.5 * (c_rv + c_rt)
        c_uv, c_ut, c_s = self.g_uv(u_v), self.g_ut(u_t), self.g_s(s)

        return {
            # representations for the recommender
            'r': r, 'u_v': u_v, 'u_t': u_t, 's': s,
            # contributions in Y space
            'c_rv': c_rv, 'c_rt': c_rt, 'c_r': c_r,
            'c_uv': c_uv, 'c_ut': c_ut, 'c_s': c_s,
            # predictions of Y
            'y_v': c_rv + c_uv,
            'y_t': c_rt + c_ut,
            'y_vt': c_r + c_uv + c_ut + c_s,
            # raw inputs for the adversaries
            'x_v': x_v, 'x_t': x_t,
        }

    # --------------------------------------------------------
    def _adv_ratio(self, adversary, x, target, w):
        """Adversary minimizes, component (through GRL) maximizes, 1 - R^2."""
        tgt = grad_reverse(target, self.cfg.grl_lambda)
        pred = adversary(x)
        mse = _per_dim_mse(pred, tgt, w)
        var = _per_dim_var(tgt, w)
        return (mse / (var + self.cfg.eps)).mean()

    def beta(self) -> float:
        c = self.cfg
        if c.beta_warmup_steps <= 0:
            return c.beta_r
        return c.beta_r * min(1.0, float(self.step.item()) / c.beta_warmup_steps)

    def compute_losses(self, out: Dict[str, torch.Tensor], Y: torch.Tensor,
                       w: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Y: [N, y_dim] standardized collaborative target (no grad).
        w: [N] item weights (0 = excluded, e.g. no train interaction / held-out).
        """
        c = self.cfg
        Y = Y.detach()
        if self.training:
            self.step += 1

        mse = lambda p: _per_dim_mse(p, Y, w).mean()

        loss_single = mse(out['y_v']) + mse(out['y_t'])
        loss_joint = mse(out['y_vt'])

        agree = _wmean((out['c_rv'] - out['c_rt']).pow(2).mean(-1), w)
        beta = self.beta()
        loss_r = mse(out['c_rv']) + mse(out['c_rt']) + beta * agree

        x_v, x_t = out['x_v'], out['x_t']
        excl_uv = self._adv_ratio(self.adv_t_uv, x_t, out['c_uv'], w)
        excl_ut = self._adv_ratio(self.adv_v_ut, x_v, out['c_ut'], w)
        excl_s = (self._adv_ratio(self.adv_v_s, x_v, out['c_s'], w)
                  + self._adv_ratio(self.adv_t_s, x_t, out['c_s'], w))
        loss_excl = excl_uv + excl_ut + excl_s

        total = (c.lambda_single * loss_single + c.lambda_joint * loss_joint
                 + c.lambda_r * loss_r + c.lambda_excl * loss_excl)
        return {
            'loss': total,
            'loss_single': loss_single.detach(),
            'loss_joint': loss_joint.detach(),
            'loss_r': loss_r.detach(),
            'agree': agree.detach(),
            'loss_excl': loss_excl.detach(),
            'beta': torch.tensor(beta),
        }

    # --------------------------------------------------------
    @torch.no_grad()
    def estimate(self, x_v: torch.Tensor, x_t: torch.Tensor, Y: torch.Tensor,
                 w: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Plug-in V-PID estimates in nats (summed over Y dims), using the
        current decoders as the (approximate) infimum over V. Evaluate on
        held-out items for an unbiased estimate.
        """
        was_training = self.training
        self.eval()
        out = self.forward(x_v, x_t)
        var = _per_dim_var(Y, w).clamp_min(1e-8)

        def info(mse_d):
            return float((0.5 * torch.log(var / mse_d.clamp_min(1e-8))).sum())

        def r2(mse_d):
            return float(1.0 - mse_d.sum() / var.sum())

        m_v = _per_dim_mse(out['y_v'], Y, w)
        m_t = _per_dim_mse(out['y_t'], Y, w)
        m_vt = _per_dim_mse(out['y_vt'], Y, w)
        m_cap = 0.5 * (_per_dim_mse(out['c_rv'], Y, w) + _per_dim_mse(out['c_rt'], Y, w))

        I_v, I_t, I_vt, R = info(m_v), info(m_t), info(m_vt), info(m_cap)
        # how much of each exclusive contribution the adversaries still recover
        def adv_r2(adv, x, tgt):
            return float(1.0 - (_per_dim_mse(adv(x), tgt, w) / (_per_dim_var(tgt, w) + self.cfg.eps)).mean())

        res = {
            'I_v': I_v, 'I_t': I_t, 'I_vt': I_vt,
            'R': R, 'U_v': I_v - R, 'U_t': I_t - R, 'S': I_vt - I_v - I_t + R,
            'R2_v': r2(m_v), 'R2_t': r2(m_t), 'R2_vt': r2(m_vt),
            'cf_only': 1.0 - r2(m_vt),
            'agree_gap': float(_wmean((out['c_rv'] - out['c_rt']).pow(2).mean(-1), w)),
            'leak_uv_from_t': adv_r2(self.adv_t_uv, x_t, out['c_uv']),
            'leak_ut_from_v': adv_r2(self.adv_v_ut, x_v, out['c_ut']),
            'leak_s_from_v': adv_r2(self.adv_v_s, x_v, out['c_s']),
            'leak_s_from_t': adv_r2(self.adv_t_s, x_t, out['c_s']),
        }
        if was_training:
            self.train()
        return res


def format_vpid(d: Dict[str, float], tag: str = '') -> str:
    head = f'[V-PID{(" " + tag) if tag else ""}] '
    return (head
            + 'I_v={I_v:.3f} I_t={I_t:.3f} I_vt={I_vt:.3f} | '
              'R={R:.3f} U_v={U_v:.3f} U_t={U_t:.3f} S={S:.3f} (nats) | '
              'R2_v={R2_v:.3f} R2_t={R2_t:.3f} R2_vt={R2_vt:.3f} cf_only={cf_only:.3f} | '
              'agree_gap={agree_gap:.4f} leak(uv<-t)={leak_uv_from_t:.3f} '
              'leak(ut<-v)={leak_ut_from_v:.3f} leak(s<-v)={leak_s_from_v:.3f} '
              'leak(s<-t)={leak_s_from_t:.3f}').format(**d)


def standardize(Y: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-6):
    mu = _wmean(Y, w)
    sd = _per_dim_var(Y, w).clamp_min(eps).sqrt()
    return (Y - mu) / sd


# ============================================================
# Synthetic sanity check with known R / U / S
# ============================================================

def _synthetic(n=6000, k=4, d_in=64, noise=0.1, seed=0):
    """
    Latent factors (each k-dim, standard normal, independent):
        a : shared by v and t        -> Y block 1 is REDUNDANT
        b : only in v                -> Y block 2 is UNIQUE-v
        c : only in t                -> Y block 3 is UNIQUE-t
        p : only in v, q : only in t -> Y block 4 = p * q is SYNERGY
            (E[p*q | p] = E[p*q | q] = 0: invisible to each modality alone)
    x_v = A_v [a, b, p, nuisance_v],  x_t = A_t [a, c, q, nuisance_t]
    """
    g = torch.Generator().manual_seed(seed)
    z = lambda: torch.randn(n, k, generator=g)
    a, b, c, p, q = z(), z(), z(), z(), z()
    nv, nt = z(), z()
    Y = torch.cat([a, b, c, p * q], dim=1)
    Y = Y + noise * torch.randn(Y.shape, generator=g)
    A_v = torch.randn(4 * k, d_in, generator=g) / (4 * k) ** 0.5
    A_t = torch.randn(4 * k, d_in, generator=g) / (4 * k) ** 0.5
    x_v = torch.tanh(torch.cat([a, b, p, nv], 1) @ A_v)
    x_t = torch.tanh(torch.cat([a, c, q, nt], 1) @ A_t)
    return x_v, x_t, Y


def main(steps: int = 4000):
    torch.manual_seed(0)
    x_v, x_t, Y = _synthetic()
    n = Y.size(0)
    tr = torch.zeros(n)
    tr[: int(0.8 * n)] = 1.0
    te = 1.0 - tr
    Y = standardize(Y, tr)

    k = Y.size(1) // 4
    cfg = VPIDConfig(visual_input_dim=x_v.size(1), text_input_dim=x_t.size(1), y_dim=Y.size(1),
                     encoder_hidden_dim=128, latent_dim=64, comp_dim=32, decoder_hidden_dim=64,
                     beta_warmup_steps=1000)
    model = PredictivePID(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(steps):
        model.train()
        out = model(x_v, x_t)
        losses = model.compute_losses(out, Y, tr)
        opt.zero_grad()
        losses['loss'].backward()
        opt.step()
        if (it + 1) % 1000 == 0:
            print(f'step {it + 1}: ' + ', '.join(f'{kk}={float(v):.4f}' for kk, v in losses.items()))
            print(format_vpid(model.estimate(x_v, x_t, Y, te), 'held-out'))

    # per-block view: which Y block each quantity comes from
    print('\nPer-block held-out estimates (expected: block1->R, block2->U_v, block3->U_t, block4->S)')
    names = ['redundant a', 'unique-v b', 'unique-t c', 'synergy p*q']
    for bi in range(4):
        sl = slice(bi * k, (bi + 1) * k)
        sub = PredictivePIDBlockView(model, sl)
        d = sub.estimate(x_v, x_t, Y, te)
        print(f'  block {bi + 1} ({names[bi]:>12}): R={d["R"]:.2f} U_v={d["U_v"]:.2f} '
              f'U_t={d["U_t"]:.2f} S={d["S"]:.2f}')


class PredictivePIDBlockView:
    """Restrict a trained model's V-PID estimate to a slice of Y dims (for the sanity check)."""
    def __init__(self, model: PredictivePID, sl: slice):
        self.model, self.sl = model, sl

    @torch.no_grad()
    def estimate(self, x_v, x_t, Y, w):
        m = self.model
        m.eval()
        out = m(x_v, x_t)
        sl = self.sl
        Ys = Y[:, sl]
        var = _per_dim_var(Ys, w).clamp_min(1e-8)
        info = lambda p: float((0.5 * torch.log(var / _per_dim_mse(p[:, sl], Ys, w).clamp_min(1e-8))).sum())
        I_v, I_t, I_vt = info(out['y_v']), info(out['y_t']), info(out['y_vt'])
        m_cap = 0.5 * (_per_dim_mse(out['c_rv'][:, sl], Ys, w) + _per_dim_mse(out['c_rt'][:, sl], Ys, w))
        R = float((0.5 * torch.log(var / m_cap.clamp_min(1e-8))).sum())
        return {'R': R, 'U_v': I_v - R, 'U_t': I_t - R, 'S': I_vt - I_v - I_t + R}


if __name__ == '__main__':
    main()
