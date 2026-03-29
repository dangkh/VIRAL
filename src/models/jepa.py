import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utilities
# =========================

def partial_mask(x, mask_ratio=0.3):
    """
    Randomly mask a portion of features in a modality.
    
    Args:
        x: tensor [B, D]
        mask_ratio: percentage of features to mask
    Returns:
        masked_x
    """
    B, D = x.shape
    
    # create random mask
    mask = (torch.rand(B, D, device=x.device) > mask_ratio).float()
    
    masked_x = x * mask
    
    return masked_x, mask

def mask_two_modalities(x1, x2, mask_ratio=0.3):

    x1_masked, mask1 = partial_mask(x1, mask_ratio)
    x2_masked, mask2 = partial_mask(x2, mask_ratio)

    return x1_masked, x2_masked, mask1, mask2


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad = requires_grad


@torch.no_grad()
def init_ema(target_model: nn.Module, online_model: nn.Module) -> None:
    """
    Initialize target model as an exact copy of online model.
    """
    for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
        p_t.data.copy_(p_o.data)
    for b_t, b_o in zip(target_model.buffers(), online_model.buffers()):
        b_t.copy_(b_o)
    set_requires_grad(target_model, False)


@torch.no_grad()
def update_ema(target_model: nn.Module, online_model: nn.Module, tau: float = 0.99) -> None:
    """
    EMA update:
        theta_target <- tau * theta_target + (1 - tau) * theta_online
    """
    for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)

    # # Keep buffers in sync. Useful if you later use BatchNorm.
    # for b_t, b_o in zip(target_model.buffers(), online_model.buffers()):
    #     b_t.copy_(b_o)


# def off_diagonal(x: torch.Tensor) -> torch.Tensor:
#     n, m = x.shape
#     if n != m:
#         raise ValueError(f"off_diagonal expects square matrix, got {x.shape}")
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, D]
    return: [D, D]
    """
    if x.dim() != 2:
        raise ValueError(f"covariance_matrix expects 2D tensor, got {x.dim()}D")
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(x.size(0) - 1, 1)
    return (x.T @ x) / denom


def cross_covariance_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Encourage x and y to be decorrelated.
    x: [B, Dx], y: [B, Dy]
    """
    if x.size(0) != y.size(0):
        raise ValueError("Batch sizes of x and y must match.")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    denom = max(x.size(0) - 1, 1)
    cov = (x.T @ y) / denom
    return (cov ** 2).mean()


def variance_regularizer(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Optional anti-collapse regularizer: encourage each dimension to keep variance.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


# =========================
# Building blocks
# =========================

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
        for i in range(num_layers - 1):
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
    Demo encoder. Replace this with your real encoder.
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
    Split latent z into:
      - redundant/shared candidate r
      - unique candidate u
    """
    def __init__(self, latent_dim: int, hidden_dim: int, comp_dim: int):
        super().__init__()
        self.to_r = MLP(latent_dim, hidden_dim, comp_dim, num_layers=2)
        self.to_u = MLP(latent_dim, hidden_dim, comp_dim, num_layers=2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.to_r(z)
        u = self.to_u(z)
        return r, u


class SynergyHead(nn.Module):
    """
    Build joint representation and synergy latent from both modalities.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, joint_dim: int, s_dim: int):
        super().__init__()
        self.joint_fuser = MLP(latent_dim * 2, hidden_dim, joint_dim, num_layers=2)
        self.to_s = MLP(joint_dim, hidden_dim, s_dim, num_layers=2)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_joint = self.joint_fuser(torch.cat([z_v, z_t], dim=-1))
        s = self.to_s(z_joint)
        return z_joint, s


class SharedPredictor(nn.Module):
    """
    Predict shared latent of one modality from the other.
    """
    def __init__(self, comp_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(comp_dim, hidden_dim, comp_dim, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SynergyPredictor(nn.Module):
    """
    Predict synergy latent from joint context.
    """
    def __init__(self, joint_dim: int, hidden_dim: int, s_dim: int):
        super().__init__()
        self.net = MLP(joint_dim, hidden_dim, s_dim, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskHead(nn.Module):
    """
    Classification head over [R, Uv, Ut, S].
    """
    def __init__(self, r_dim: int, u_dim: int, s_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        in_dim = r_dim + u_dim + u_dim + s_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, r: torch.Tensor, u_v: torch.Tensor, u_t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([r, u_v, u_t, s], dim=-1)
        return self.net(x)


# =========================
# Config
# =========================

@dataclass
class PIDJEPAConfig:
    visual_input_dim: int = 128
    text_input_dim: int = 128

    encoder_hidden_dim: int = 256
    latent_dim: int = 128

    decomp_hidden_dim: int = 128
    comp_dim: int = 64

    synergy_hidden_dim: int = 128
    joint_dim: int = 128
    s_dim: int = 64

    predictor_hidden_dim: int = 128
    task_hidden_dim: int = 128
    num_classes: int = 6

    lambda_r: float = 1.0
    lambda_u: float = 0.1
    lambda_s: float = 1.0
    lambda_sep_s: float = 0.1
    lambda_var: float = 0.01

    ema_tau: float = 0.99
    
    mask_ratio: float = 0.1


# =========================
# PID-JEPA Model
# =========================

class PIDJEPA(nn.Module):
    def __init__(self, cfg = None):
        super().__init__()
        if cfg is None:
            cfg = PIDJEPAConfig()
        self.cfg = cfg

        # Online branch
        self.visual_encoder = SimpleEncoder(
            cfg.visual_input_dim, cfg.encoder_hidden_dim, cfg.latent_dim
        )
        self.text_encoder = SimpleEncoder(
            cfg.text_input_dim, cfg.encoder_hidden_dim, cfg.latent_dim
        )

        self.visual_decomp = DecompositionHead(
            cfg.latent_dim, cfg.decomp_hidden_dim, cfg.comp_dim
        )
        self.text_decomp = DecompositionHead(
            cfg.latent_dim, cfg.decomp_hidden_dim, cfg.comp_dim
        )

        self.synergy_head = SynergyHead(
            cfg.latent_dim, cfg.synergy_hidden_dim, cfg.joint_dim, cfg.s_dim
        )

        self.pred_v_to_t = SharedPredictor(cfg.comp_dim, cfg.predictor_hidden_dim)
        self.pred_t_to_v = SharedPredictor(cfg.comp_dim, cfg.predictor_hidden_dim)
        self.pred_s = SynergyPredictor(cfg.joint_dim, cfg.predictor_hidden_dim, cfg.s_dim)

        self.task_head = TaskHead(
            r_dim=cfg.comp_dim,
            u_dim=cfg.comp_dim,
            s_dim=cfg.s_dim,
            hidden_dim=cfg.task_hidden_dim,
            num_classes=cfg.num_classes,
        )

        # Target branch: EMA copies
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

    @torch.no_grad()
    def update_target(self) -> None:
        tau = self.cfg.ema_tau
        update_ema(self.visual_encoder_t, self.visual_encoder, tau)
        update_ema(self.text_encoder_t, self.text_encoder, tau)
        update_ema(self.visual_decomp_t, self.visual_decomp, tau)
        update_ema(self.text_decomp_t, self.text_decomp, tau)
        update_ema(self.synergy_head_t, self.synergy_head, tau)

    def encode_online(
        self, x_v_ctx: torch.Tensor, x_t_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_v = self.visual_encoder(x_v_ctx)
        z_t = self.text_encoder(x_t_ctx)

        r_v, u_v = self.visual_decomp(z_v)
        r_t, u_t = self.text_decomp(z_t)

        z_joint, s = self.synergy_head(z_v, z_t)

        return {
            "z_v": z_v,
            "z_t": z_t,
            "r_v": r_v,
            "u_v": u_v,
            "r_t": r_t,
            "u_t": u_t,
            "z_joint": z_joint,
            "s": s,
        }

    @torch.no_grad()
    def encode_target(
        self, x_v_full: torch.Tensor, x_t_full: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_v = self.visual_encoder_t(x_v_full)
        z_t = self.text_encoder_t(x_t_full)

        r_v, u_v = self.visual_decomp_t(z_v)
        r_t, u_t = self.text_decomp_t(z_t)

        z_joint, s = self.synergy_head_t(z_v, z_t)

        return {
            "z_v": z_v,
            "z_t": z_t,
            "r_v": r_v,
            "u_v": u_v,
            "r_t": r_t,
            "u_t": u_t,
            "z_joint": z_joint,
            "s": s,
        }

    def forward(
        self,
        x_v_full: torch.Tensor,
        x_t_full: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Create masked context inputs
        x_v_ctx, x_t_ctx, _, _ = mask_two_modalities(x_v_full, x_t_full, mask_ratio=self.cfg.mask_ratio)
        online = self.encode_online(x_v_ctx, x_t_ctx)
        target = self.encode_target(x_v_full, x_t_full)

        # Shared JEPA predictions
        r_t_hat = self.pred_v_to_t(online["r_v"])
        r_v_hat = self.pred_t_to_v(online["r_t"])

        # Shared fused representation
        r = 0.5 * (online["r_v"] + online["r_t"])

        # Synergy JEPA prediction
        s_hat = self.pred_s(online["z_joint"])

        # Task logits
        logits = self.task_head(r, online["u_v"], online["u_t"], online["s"])

        return {
            **online,
            "target_r_v": target["r_v"].detach(),
            "target_r_t": target["r_t"].detach(),
            "target_s": target["s"].detach(),
            "r_v_hat": r_v_hat,
            "r_t_hat": r_t_hat,
            "r": r,
            "s_hat": s_hat,
            "logits": logits,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        # -------------------------
        # 1) Task loss
        # -------------------------
        # with rec task loss

        # -------------------------
        # 2) R loss: cross-modal JEPA
        # -------------------------
        loss_r = F.mse_loss(outputs["r_t_hat"], outputs["target_r_t"]) + \
                 F.mse_loss(outputs["r_v_hat"], outputs["target_r_v"])

        # -------------------------
        # 3) U loss: keep unique separate from shared and opposite modality
        # -------------------------
        # Separate U from shared R
        loss_u_sep_shared = cross_covariance_penalty(outputs["u_v"], outputs["r"]) + \
                            cross_covariance_penalty(outputs["u_t"], outputs["r"])

        # Separate U from opposite modality latent
        loss_u_cross = cross_covariance_penalty(outputs["u_v"], outputs["z_t"]) + \
                       cross_covariance_penalty(outputs["u_t"], outputs["z_v"])

        loss_u = loss_u_sep_shared + loss_u_cross

        # -------------------------
        # 4) S loss: joint JEPA
        # -------------------------
        loss_s = F.mse_loss(outputs["s_hat"], outputs["target_s"])

        # -------------------------
        # 5) Separate S from R and U
        # -------------------------
        loss_sep_s = cross_covariance_penalty(outputs["s"], outputs["r"]) + \
                     cross_covariance_penalty(outputs["s"], outputs["u_v"]) + \
                     cross_covariance_penalty(outputs["s"], outputs["u_t"])

        # -------------------------
        # 6) Optional anti-collapse variance regularizer
        # -------------------------
        loss_var = (
            variance_regularizer(outputs["r_v"]) +
            variance_regularizer(outputs["r_t"]) +
            variance_regularizer(outputs["u_v"]) +
            variance_regularizer(outputs["u_t"]) +
            variance_regularizer(outputs["s"])
        )

        # -------------------------
        # 7) Total
        # -------------------------
        total = (
             cfg.lambda_r * loss_r
            + cfg.lambda_u * loss_u
            + cfg.lambda_s * loss_s
            + cfg.lambda_sep_s * loss_sep_s
            + cfg.lambda_var * loss_var
        )

        return {
            "loss": total,
            "loss_r": loss_r,
            "loss_u": loss_u,
            "loss_s": loss_s,
            "loss_sep_s": loss_sep_s,
            "loss_var": loss_var,
        }


# =========================
# Masking helpers
# =========================

def random_feature_mask(x: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor:
    """
    Simple feature masking for demo.
    x: [B, D]
    """
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in [0, 1)")
    if x.dim() != 2:
        raise ValueError(f"random_feature_mask expects [B, D], got {x.shape}")

    keep_prob = 1.0 - mask_ratio
    mask = torch.bernoulli(torch.full_like(x, keep_prob))
    return x * mask


# =========================
# Example training step
# =========================

def train_step(
    model: PIDJEPA,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """
    batch should contain:
      - x_v: [B, Dv]
      - x_t: [B, Dt]
      - y:   [B]
    """
    model.train()

    x_v = batch["x_v"].to(device)
    x_t = batch["x_t"].to(device)
    y = batch["y"].to(device)

    # Context inputs: masked
    x_v_ctx = random_feature_mask(x_v, mask_ratio=0.3)
    x_t_ctx = random_feature_mask(x_t, mask_ratio=0.3)

    # Target inputs: full
    x_v_full = x_v
    x_t_full = x_t

    outputs = model(
        x_v_ctx=x_v_ctx,
        x_t_ctx=x_t_ctx,
        x_v_full=x_v_full,
        x_t_full=x_t_full,
    )
    losses = model.compute_losses(outputs, y)

    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()

    # EMA update after optimizer step
    model.update_target()

    return {k: float(v.detach().cpu()) for k, v in losses.items()}


# =========================
# Example usage
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        task_hidden_dim=64,
        num_classes=6,
        lambda_r=1.0,
        lambda_u=0.1,
        lambda_s=1.0,
        lambda_sep_s=0.1,
        lambda_var=0.01,
        ema_tau=0.99,
    )

    model = PIDJEPA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dummy batch
    batch = {
        "x_v": torch.randn(32, cfg.visual_input_dim),
        "x_t": torch.randn(32, cfg.text_input_dim),
        "y": torch.randint(0, cfg.num_classes, (32,)),
    }

    for step in range(5):
        stats = train_step(model, optimizer, batch, device)
        print(f"Step {step}: {stats}")


if __name__ == "__main__":
    main()