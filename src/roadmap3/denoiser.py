"""D2: Transformer denoiser over an unordered cell set -- ROADMAP3's
alternative to Phase5/6's patch UNet and ROADMAP2 R2's GNN. Full
self-attention across all tokens (no causal mask, no fixed edges: this is a
set, not a sequence or a graph with prescribed adjacency).

Architecture is DiT-style (Peebles & Xie, "Scalable Diffusion Models with
Transformers"): diffusion-timestep conditioning enters via AdaLN-Zero
(global, one value per batch item, modulates every block, gated at zero
init so each block starts as an identity map -- stabilizes early training).

Per-token spatial/control conditioning (the ControlNet-like part -- size
gradient, alignment flow, sampled at each cell's own position, see
control_fields.py) is different in kind from the timestep: it varies *per
token*, not per batch item, so it can't go through the same global AdaLN
path. It's added directly into each token's embedding instead, the same
role a ControlNet side-branch's per-pixel feature map plays for a UNet.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] (int or float). Returns [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _zero_init(module: nn.Linear):
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class AdaLNZeroBlock(nn.Module):
    """Pre-norm self-attention + MLP, both modulated/gated by a per-batch-item
    conditioning vector (the diffusion timestep embedding). Gate projections
    are zero-initialized (DiT's trick) so at init this block is the identity
    -- the model starts as a no-op and only gradually learns to use each
    layer, rather than fighting a randomly-scaled residual from step one."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = _zero_init(nn.Linear(cond_dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: torch.Tensor):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out

        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class SetDenoiser(nn.Module):
    """Predicts the noise added to a [B, N, attr_dim] set of cell attribute
    vectors, given the diffusion timestep and per-token control-field
    conditioning. N is not fixed by the architecture -- padding to a common
    N_max (dataset.py) is only a training-batch convenience."""

    def __init__(self, attr_dim: int, ctrl_dim: int, hidden_dim: int = 128,
                 n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(attr_dim, hidden_dim)
        self.ctrl_proj = nn.Linear(ctrl_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(hidden_dim, n_heads, cond_dim=hidden_dim) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_ada = _zero_init(nn.Linear(hidden_dim, 2 * hidden_dim))
        self.out_proj = _zero_init(nn.Linear(hidden_dim, attr_dim))

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, ctrl: torch.Tensor,
                active_mask: torch.Tensor) -> torch.Tensor:
        """x_noisy: [B, N, attr_dim], t: [B], ctrl: [B, N, ctrl_dim],
        active_mask: [B, N] (1 = real token, 0 = pad)."""
        h = self.input_proj(x_noisy) + self.ctrl_proj(ctrl)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.hidden_dim))

        key_padding_mask = active_mask < 0.5  # True = ignore, per nn.MultiheadAttention convention
        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask)

        shift, scale = self.final_ada(t_emb).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(h)
