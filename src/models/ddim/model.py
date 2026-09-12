"""Small UNet with FiLM conditioning on the Phase 3 GVAE composition latent,
per PLAN.md Phase 5: "start with a small model, short training run to prove
the loop works end-to-end before scaling."
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, twice, with a residual connection and a
    FiLM (scale+shift) modulation from the timestep+conditioning embedding
    applied between the two convs -- the mechanism the plan specifies for
    injecting the Phase 3 latent (vs. cross-attention, simpler and enough
    for a small proof-of-concept model)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class FieldUNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int | None = None, cond_dim: int = 32,
                 base_ch: int = 32, time_emb_dim: int = 128):
        """`out_channels` defaults to `in_channels` (the original, everything-
        is-denoised setup). Pass it explicitly smaller than `in_channels`
        when some input channels are clean spatial conditioning (e.g. a
        boundary_sdf crop) concatenated alongside the noisy target rather
        than something the model is meant to reconstruct -- those channels
        are still consumed by the down/up path, just never predicted."""
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.time_emb_dim = time_emb_dim
        emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        self.stem = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Downsample: 128 -> 64 -> 32
        self.down1 = FiLMResBlock(base_ch, base_ch, emb_dim)
        self.down_pool1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.down2 = FiLMResBlock(base_ch * 2, base_ch * 2, emb_dim)
        self.down_pool2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)

        self.mid = FiLMResBlock(base_ch * 4, base_ch * 4, emb_dim)

        # Upsample: 32 -> 64 -> 128, with skip connections
        self.up_conv2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.up2 = FiLMResBlock(base_ch * 4, base_ch * 2, emb_dim)  # concat skip
        self.up_conv1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.up1 = FiLMResBlock(base_ch * 2, base_ch, emb_dim)  # concat skip

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_channels, 3, padding=1)

    def forward(self, x, timesteps, cond):
        t_emb = self.time_mlp(sinusoidal_embedding(timesteps, self.time_emb_dim))
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb

        h0 = self.stem(x)
        h1 = self.down1(h0, emb)
        h1d = self.down_pool1(h1)
        h2 = self.down2(h1d, emb)
        h2d = self.down_pool2(h2)

        m = self.mid(h2d, emb)

        u2 = self.up_conv2(m)
        u2 = self.up2(torch.cat([u2, h2], dim=1), emb)
        u1 = self.up_conv1(u2)
        u1 = self.up1(torch.cat([u1, h1], dim=1), emb)

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


if __name__ == "__main__":
    model = FieldUNet()
    x = torch.randn(2, 4, 128, 128)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 32)
    out = model(x, t, cond)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {out.shape}, params: {n_params:,}")
