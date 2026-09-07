import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def partial_mask(x: torch.Tensor, mask_ratio: float = 0.2):
    """
    Feature-wise random masking.

    Args:
        x: [B, D]
        mask_ratio: fraction of dimensions to mask.

    Returns:
        x_masked: [B, D]
        mask: [B, D], 1 = kept, 0 = masked
    """
    if x.dim() != 2:
        raise ValueError(f"partial_mask expects [B, D], got {tuple(x.shape)}")
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in [0, 1).")

    mask = (torch.rand_like(x) > mask_ratio).to(x.dtype)
    return x * mask, mask


def mask_two_modalities(
    x_v: torch.Tensor,
    x_t: torch.Tensor,
    mask_ratio: float = 0.2,
):
    x_v_ctx, mask_v = partial_mask(x_v, mask_ratio)
    x_t_ctx, mask_t = partial_mask(x_t, mask_ratio)
    return x_v_ctx, x_t_ctx, mask_v, mask_t


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = requires_grad


@torch.no_grad()
def init_ema(target_model: nn.Module, online_model: nn.Module) -> None:
    target_model.load_state_dict(online_model.state_dict(), strict=True)
    set_requires_grad(target_model, False)


@torch.no_grad()
def update_ema(
    target_model: nn.Module,
    online_model: nn.Module,
    tau: float = 0.99,
) -> None:
    """
    theta_target <- tau * theta_target + (1 - tau) * theta_online
    """
    for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

    # Keep buffers aligned (important if BN is later used).
    for b_t, b_o in zip(target_model.buffers(), online_model.buffers()):
        b_t.copy_(b_o)


def cross_covariance_penalty(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize linear dependence between two representations.

    x: [B, Dx]
    y: [B, Dy]
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("cross_covariance_penalty expects 2D tensors.")
    if x.size(0) != y.size(0):
        raise ValueError("Batch sizes must match.")

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    denom = max(x.size(0) - 1, 1)
    cov = (x.T @ y) / denom
    return cov.pow(2).mean()


def variance_regularizer(
    x: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg-style anti-collapse term.
    """
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


# ============================================================
# Gradient reversal for synergy exclusivity
# ============================================================

class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0):
    """
    Forward: identity.
    Backward to x: multiply gradient by -lambda.

    This lets the single-modality predictor learn to predict S,
    while the online encoder learns to make S difficult to predict
    from a single modality.
    """
    return _GradientReversalFn.apply(x, lambd)


# ============================================================
# Building blocks
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        d_in = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim

        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleEncoder(nn.Module):
    """
    Replace with the real modality encoder if needed.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecompositionHead(nn.Module):
    """
    From a modality latent z, produce:
      r: redundant/shared candidate
      u: modality-unique candidate
    """
    def __init__(self, latent_dim: int, hidden_dim: int, comp_dim: int):
        super().__init__()
        self.to_r = MLP(latent_dim, hidden_dim, comp_dim, num_layers=2)
        self.to_u = MLP(latent_dim, hidden_dim, comp_dim, num_layers=2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.to_r(z), self.to_u(z)


class SynergyHead(nn.Module):
    """
    IMPORTANT:
    Synergy is built from encoded modality latents z_v and z_t,
    not directly from raw input features.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        joint_dim: int,
        s_dim: int,
    ):
        super().__init__()
        self.joint_fuser = MLP(
            latent_dim * 2,
            hidden_dim,
            joint_dim,
            num_layers=2,
        )
        self.to_s = MLP(
            joint_dim,
            hidden_dim,
            s_dim,
            num_layers=2,
        )

    def forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_joint = self.joint_fuser(torch.cat([z_v, z_t], dim=-1))
        s = self.to_s(z_joint)
        return z_joint, s


class Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = MLP(in_dim, hidden_dim, out_dim, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PIDJEPAConfig:
    visual_input_dim: int = 384
    text_input_dim: int = 384

    encoder_hidden_dim: int = 64
    latent_dim: int = 64

    decomp_hidden_dim: int = 64
    comp_dim: int = 64

    synergy_hidden_dim: int = 64
    joint_dim: int = 64
    s_dim: int = 64

    predictor_hidden_dim: int = 64

    # JEPA/PID losses
    lambda_r: float = 1.0

    # Unique = own-modality JEPA + exclusion constraints
    lambda_u_pred: float = 1.0
    lambda_u_sep_r: float = 0.1
    lambda_u_cross: float = 0.1

    # Synergy = joint JEPA + adversarial single-modality exclusion
    lambda_s_joint: float = 1.0
    lambda_s_single_adv: float = 0.1
    grl_lambda: float = 1.0

    # Component separation
    lambda_s_sep: float = 0.1

    # Anti-collapse
    lambda_var: float = 0.01

    # EMA
    ema_tau: float = 0.99

    # Context corruption
    mask_ratio: float = 0.2


# ============================================================
# PID-JEPA
# ============================================================

class PIDJEPA(nn.Module):
    """
    JEPA operationalization of PID:

    R:
        masked V -> full target R_T
        masked T -> full target R_V

    U_v:
        masked V -> full target U_V
        while U_V is decorrelated from R and T

    U_t:
        masked T -> full target U_T
        while U_T is decorrelated from R and V

    S:
        masked (V,T) -> full target S
        while adversarial single-modality predictors try to recover S
        and the modality encoders are trained (via GRL) to prevent that.

    Thus:
        R  = cross-modal predictable
        U  = own-modal predictable, cross-modal excluded
        S  = joint predictable, single-modal discouraged
    """

    def __init__(self, cfg: PIDJEPAConfig = None):
        super().__init__()

        self.cfg = cfg or PIDJEPAConfig()
        cfg = self.cfg

        # -------------------------
        # Online encoders
        # -------------------------
        self.visual_encoder = SimpleEncoder(
            cfg.visual_input_dim,
            cfg.encoder_hidden_dim,
            cfg.latent_dim,
        )
        self.text_encoder = SimpleEncoder(
            cfg.text_input_dim,
            cfg.encoder_hidden_dim,
            cfg.latent_dim,
        )

        # -------------------------
        # Online PID heads
        # -------------------------
        self.visual_decomp = DecompositionHead(
            cfg.latent_dim,
            cfg.decomp_hidden_dim,
            cfg.comp_dim,
        )
        self.text_decomp = DecompositionHead(
            cfg.latent_dim,
            cfg.decomp_hidden_dim,
            cfg.comp_dim,
        )

        self.synergy_head = SynergyHead(
            cfg.latent_dim,
            cfg.synergy_hidden_dim,
            cfg.joint_dim,
            cfg.s_dim,
        )

        # -------------------------
        # R predictors:
        # cross-modal JEPA
        # -------------------------
        self.pred_r_v_to_t = Predictor(
            cfg.comp_dim,
            cfg.predictor_hidden_dim,
            cfg.comp_dim,
        )
        self.pred_r_t_to_v = Predictor(
            cfg.comp_dim,
            cfg.predictor_hidden_dim,
            cfg.comp_dim,
        )

        # -------------------------
        # U predictors:
        # own-modality masked-to-full JEPA
        # -------------------------
        self.pred_u_v = Predictor(
            cfg.comp_dim,
            cfg.predictor_hidden_dim,
            cfg.comp_dim,
        )
        self.pred_u_t = Predictor(
            cfg.comp_dim,
            cfg.predictor_hidden_dim,
            cfg.comp_dim,
        )

        # -------------------------
        # S predictor:
        # joint masked-to-full JEPA
        # -------------------------
        self.pred_s_joint = Predictor(
            cfg.joint_dim,
            cfg.predictor_hidden_dim,
            cfg.s_dim,
        )

        # -------------------------
        # Single-modality adversaries for S
        # These predictors MINIMIZE prediction error.
        # GRL makes modality encoders MAXIMIZE it.
        # -------------------------
        self.pred_s_from_v = Predictor(
            cfg.latent_dim,
            cfg.predictor_hidden_dim,
            cfg.s_dim,
        )
        self.pred_s_from_t = Predictor(
            cfg.latent_dim,
            cfg.predictor_hidden_dim,
            cfg.s_dim,
        )

        # -------------------------
        # EMA target networks
        # -------------------------
        self.visual_encoder_t = copy.deepcopy(self.visual_encoder)
        self.text_encoder_t = copy.deepcopy(self.text_encoder)

        self.visual_decomp_t = copy.deepcopy(self.visual_decomp)
        self.text_decomp_t = copy.deepcopy(self.text_decomp)

        self.synergy_head_t = copy.deepcopy(self.synergy_head)

        init_ema(self.visual_encoder_t, self.visual_encoder)
        init_ema(self.text_encoder_t, self.text_encoder)
        init_ema(self.visual_decomp_t, self.visual_decomp)
        init_ema(self.text_decomp_t, self.text_decomp)
        init_ema(self.synergy_head_t, self.synergy_head)

    # --------------------------------------------------------
    # EMA update
    # --------------------------------------------------------

    @torch.no_grad()
    def update_target(self) -> None:
        tau = self.cfg.ema_tau

        update_ema(self.visual_encoder_t, self.visual_encoder, tau)
        update_ema(self.text_encoder_t, self.text_encoder, tau)

        update_ema(self.visual_decomp_t, self.visual_decomp, tau)
        update_ema(self.text_decomp_t, self.text_decomp, tau)

        update_ema(self.synergy_head_t, self.synergy_head, tau)

    # --------------------------------------------------------
    # Online / target encoding
    # --------------------------------------------------------

    def encode_online(
        self,
        x_v_ctx: torch.Tensor,
        x_t_ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        z_v = self.visual_encoder(x_v_ctx)
        z_t = self.text_encoder(x_t_ctx)

        r_v, u_v = self.visual_decomp(z_v)
        r_t, u_t = self.text_decomp(z_t)

        # Corrected: use encoded latents, not raw x_v/x_t.
        z_joint, s = self.synergy_head(z_v, z_t)

        return {
            "z_v": z_v,
            "z_t": z_t,
            "r_v": r_v,
            "r_t": r_t,
            "u_v": u_v,
            "u_t": u_t,
            "z_joint": z_joint,
            "s": s,
        }

    @torch.no_grad()
    def encode_target(
        self,
        x_v_full: torch.Tensor,
        x_t_full: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        z_v = self.visual_encoder_t(x_v_full)
        z_t = self.text_encoder_t(x_t_full)

        r_v, u_v = self.visual_decomp_t(z_v)
        r_t, u_t = self.text_decomp_t(z_t)

        z_joint, s = self.synergy_head_t(z_v, z_t)

        return {
            "z_v": z_v,
            "z_t": z_t,
            "r_v": r_v,
            "r_t": r_t,
            "u_v": u_v,
            "u_t": u_t,
            "z_joint": z_joint,
            "s": s,
        }

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x_v_full: torch.Tensor,
        x_t_full: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Mask exactly ONCE, here.
        x_v_ctx, x_t_ctx, mask_v, mask_t = mask_two_modalities(
            x_v_full,
            x_t_full,
            mask_ratio=self.cfg.mask_ratio,
        )

        online = self.encode_online(x_v_ctx, x_t_ctx)
        target = self.encode_target(x_v_full, x_t_full)

        # Full ONLINE single-modality latents are used only for
        # exclusivity tests. This is important:
        # PID asks whether S/U can be explained by the FULL
        # opposite/single modality, not merely by a masked view.
        z_v_full_online = self.visual_encoder(x_v_full)
        z_t_full_online = self.text_encoder(x_t_full)

        # ====================================================
        # R: cross-modal JEPA
        # ====================================================
        r_t_hat = self.pred_r_v_to_t(online["r_v"])
        r_v_hat = self.pred_r_t_to_v(online["r_t"])

        # Shared representation used by downstream recommendation.
        r = 0.5 * (online["r_v"] + online["r_t"])
        r_target = 0.5 * (target["r_v"] + target["r_t"])

        # ====================================================
        # U: own-modality masked-to-full JEPA
        # ====================================================
        u_v_hat = self.pred_u_v(online["u_v"])
        u_t_hat = self.pred_u_t(online["u_t"])

        # ====================================================
        # S: joint JEPA
        # ====================================================
        s_hat = self.pred_s_joint(online["z_joint"])

        # ====================================================
        # S: single-modality adversaries
        #
        # Predictor learns: z_v -> target_s / z_t -> target_s.
        # Encoder receives reversed gradient and therefore tries
        # to make target synergy unavailable from one modality.
        # ====================================================
        z_v_grl = grad_reverse(
            z_v_full_online,
            lambd=self.cfg.grl_lambda,
        )
        z_t_grl = grad_reverse(
            z_t_full_online,
            lambd=self.cfg.grl_lambda,
        )

        s_from_v_hat = self.pred_s_from_v(z_v_grl)
        s_from_t_hat = self.pred_s_from_t(z_t_grl)

        return {
            **online,

            # masks
            "mask_v": mask_v,
            "mask_t": mask_t,

            # full online modality latents for exclusivity tests
            "z_v_full_online": z_v_full_online,
            "z_t_full_online": z_t_full_online,

            # downstream representations
            "r": r,
            "rF": r_target.detach(),

            # EMA targets
            "target_r_v": target["r_v"].detach(),
            "target_r_t": target["r_t"].detach(),
            "target_u_v": target["u_v"].detach(),
            "target_u_t": target["u_t"].detach(),
            "target_s": target["s"].detach(),

            # JEPA predictions
            "r_v_hat": r_v_hat,
            "r_t_hat": r_t_hat,
            "u_v_hat": u_v_hat,
            "u_t_hat": u_t_hat,
            "s_hat": s_hat,

            # single-modality S adversaries
            "s_from_v_hat": s_from_v_hat,
            "s_from_t_hat": s_from_t_hat,
        }

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        cfg = self.cfg

        # ====================================================
        # 1) REDUNDANCY
        #
        # V_context -> R_T(full)
        # T_context -> R_V(full)
        # ====================================================
        loss_r = (
            F.mse_loss(
                outputs["r_t_hat"],
                outputs["target_r_t"],
            )
            +
            F.mse_loss(
                outputs["r_v_hat"],
                outputs["target_r_v"],
            )
        )

        # ====================================================
        # 2) UNIQUENESS: positive own-modality information
        #
        # masked V -> U_V(full)
        # masked T -> U_T(full)
        #
        # This fixes the previous issue where U was only
        # decorrelated but had no positive information objective.
        # ====================================================
        loss_u_pred = (
            F.mse_loss(
                outputs["u_v_hat"],
                outputs["target_u_v"],
            )
            +
            F.mse_loss(
                outputs["u_t_hat"],
                outputs["target_u_t"],
            )
        )

        # U should not simply duplicate R.
        loss_u_sep_r = (
            cross_covariance_penalty(
                outputs["u_v"],
                outputs["r"].detach(),
            )
            +
            cross_covariance_penalty(
                outputs["u_t"],
                outputs["r"].detach(),
            )
        )

        # U_V should be difficult to explain by FULL T,
        # U_T should be difficult to explain by FULL V.
        #
        # Detach the reference modality so this exclusion loss
        # shapes U rather than degrading the opposite encoder.
        loss_u_cross = (
            cross_covariance_penalty(
                outputs["u_v"],
                outputs["z_t_full_online"].detach(),
            )
            +
            cross_covariance_penalty(
                outputs["u_t"],
                outputs["z_v_full_online"].detach(),
            )
        )

        # ====================================================
        # 3) SYNERGY: positive joint information
        #
        # masked (V,T) -> S(full V,T)
        # ====================================================
        loss_s_joint = F.mse_loss(
            outputs["s_hat"],
            outputs["target_s"],
        )

        # ====================================================
        # 4) SYNERGY: single-modality exclusion
        #
        # IMPORTANT:
        # We MINIMIZE this loss normally.
        # - predictor params learn to recover S
        # - due to GRL, encoder receives reversed gradient
        #   and learns to REMOVE single-modality predictability.
        # ====================================================
        loss_s_single_adv = (
            F.mse_loss(
                outputs["s_from_v_hat"],
                outputs["target_s"],
            )
            +
            F.mse_loss(
                outputs["s_from_t_hat"],
                outputs["target_s"],
            )
        )

        # S should be separated from R and U.
        loss_s_sep = (
            cross_covariance_penalty(
                outputs["s"],
                outputs["r"].detach(),
            )
            +
            cross_covariance_penalty(
                outputs["s"],
                outputs["u_v"].detach(),
            )
            +
            cross_covariance_penalty(
                outputs["s"],
                outputs["u_t"].detach(),
            )
        )

        # ====================================================
        # 5) Anti-collapse
        # ====================================================
        loss_var = (
            variance_regularizer(outputs["r_v"])
            +
            variance_regularizer(outputs["r_t"])
            +
            variance_regularizer(outputs["u_v"])
            +
            variance_regularizer(outputs["u_t"])
            +
            variance_regularizer(outputs["s"])
        )

        # ====================================================
        # 6) Total JEPA-PID loss
        # ====================================================
        total = (
            cfg.lambda_r * loss_r

            + cfg.lambda_u_pred * loss_u_pred
            + cfg.lambda_u_sep_r * loss_u_sep_r
            + cfg.lambda_u_cross * loss_u_cross

            + cfg.lambda_s_joint * loss_s_joint
            + cfg.lambda_s_single_adv * loss_s_single_adv
            + cfg.lambda_s_sep * loss_s_sep

            + cfg.lambda_var * loss_var
        )

        return {
            "loss": total,

            "loss_r": loss_r,

            "loss_u_pred": loss_u_pred,
            "loss_u_sep_r": loss_u_sep_r,
            "loss_u_cross": loss_u_cross,

            "loss_s_joint": loss_s_joint,
            "loss_s_single_adv": loss_s_single_adv,
            "loss_s_sep": loss_s_sep,

            "loss_var": loss_var,
        }


# ============================================================
# Example training step
# ============================================================

def train_step(
    model: PIDJEPA,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    batch:
        x_v: [B, D_v]
        x_t: [B, D_t]

    Recommendation/task loss should normally be added in the
    parent recommender model using outputs["r"], outputs["u_v"],
    outputs["u_t"], outputs["s"].
    """
    model.train()

    x_v = batch["x_v"].to(device)
    x_t = batch["x_t"].to(device)

    # No masking here: forward() owns the context corruption.
    outputs = model(
        x_v_full=x_v,
        x_t_full=x_t,
    )

    losses = model.compute_losses(outputs)

    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()

    # JEPA target update happens after online optimization.
    model.update_target()

    return {
        k: float(v.detach().cpu())
        for k, v in losses.items()
    }


# ============================================================
# Minimal sanity check
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg = PIDJEPAConfig(
        visual_input_dim=384,
        text_input_dim=384,

        encoder_hidden_dim=64,
        latent_dim=64,

        decomp_hidden_dim=64,
        comp_dim=64,

        synergy_hidden_dim=64,
        joint_dim=64,
        s_dim=64,

        predictor_hidden_dim=64,

        lambda_r=1.0,

        lambda_u_pred=1.0,
        lambda_u_sep_r=0.1,
        lambda_u_cross=0.1,

        lambda_s_joint=1.0,
        lambda_s_single_adv=0.1,
        grl_lambda=1.0,

        lambda_s_sep=0.1,

        lambda_var=0.01,

        ema_tau=0.99,
        mask_ratio=0.2,
    )

    model = PIDJEPA(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    batch = {
        "x_v": torch.randn(32, cfg.visual_input_dim),
        "x_t": torch.randn(32, cfg.text_input_dim),
    }

    for step in range(5):
        stats = train_step(
            model,
            optimizer,
            batch,
            device,
        )
        print(f"Step {step}: {stats}")


if __name__ == "__main__":
    main()
