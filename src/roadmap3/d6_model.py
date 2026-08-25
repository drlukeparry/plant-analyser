"""D6 prototype: a small conv encoder-decoder predicting a local density
field from a boundary mask alone -- input [1, H, W] (boundary_mask),
output [1, H, W] (log-density). Deliberately small and plain (no attention,
no diffusion) per this repo's "start small" precedent -- this is a
regression task (reconstruct a real, smooth, low-frequency spatial pattern
from a silhouette), not something that obviously needs a generative model
on a first pass. If a plain regressor turns out inadequate (predicts only
the boundary-averaged density, ignores real spatial structure like the
pith-center dip found in D6's sanity check), a conditional/diffusion
version would be the natural next step -- not assumed necessary yet.
"""
import torch.nn as nn


class DensityFieldNet(nn.Module):
    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.SiLU(),
            )

        self.down1 = block(1, c)
        self.down2 = block(c, c * 2, stride=2)
        self.down3 = block(c * 2, c * 4, stride=2)
        self.mid = block(c * 4, c * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), block(c * 4, c * 2))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), block(c * 2, c))
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, boundary_mask):
        h1 = self.down1(boundary_mask)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h = self.mid(h3)
        h = self.up2(h) + h2
        h = self.up1(h) + h1
        return self.out(h)
