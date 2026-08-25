"""D6 prototype training loop: reconstruct each real image's own density
raster from its boundary mask alone. Only 213 real images total (one
raster pair per image, not the effectively-unlimited augmented crops the
token-DDIM's subgraph sampling could draw) -- random flip/90-degree
rotation augmentation is used every step since a density field has genuine
rotational/reflective symmetry (unlike the graph case, this is legitimate,
cheap, and meaningfully increases effective examples).

Trains in log1p(density) space -- D6's sanity check found raw density
scale varies ~5x between images (3,500 vs 17,500 peak), the same kind of
cross-image scale leak Phase 3 already diagnosed and fixed for node
features; log-space compresses that before it can dominate the loss.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.models.gvae.train import device
from src.roadmap3.d6_density_field import GRID_RES, rasterize_image
from src.roadmap3.d6_model import DensityFieldNet
from src.roadmap3.train import OUT_DIR


def load_rasters(tag: str = "norm"):
    graph_dir = out_dir_for(tag)
    masks, densities = [], []
    for p in sorted(graph_dir.glob("*.pt")):
        cached = torch.load(p, weights_only=False)
        mask, density = rasterize_image(cached["node_df"])
        masks.append(mask)
        densities.append(density)
    return np.stack(masks), np.stack(densities)


def augment(mask: torch.Tensor, density: torch.Tensor):
    k = torch.randint(0, 4, (1,)).item()
    mask, density = torch.rot90(mask, k, dims=(-2, -1)), torch.rot90(density, k, dims=(-2, -1))
    if torch.rand(1).item() < 0.5:
        mask, density = torch.flip(mask, dims=(-1,)), torch.flip(density, dims=(-1,))
    return mask, density


def train(steps: int = 1500, batch_size: int = 8, lr: float = 1e-3, base_ch: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    dev = device()
    print("rasterizing all images ...")
    masks, densities = load_rasters()
    n = masks.shape[0]
    print(f"{n} images, grid {GRID_RES}x{GRID_RES}")

    log_density = np.log1p(densities)
    masks_t = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
    log_density_t = torch.tensor(log_density, dtype=torch.float32).unsqueeze(1)

    model = DensityFieldNet(base_ch=base_ch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: base_ch={base_ch}, {n_params:,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        mb, db = masks_t[idx].to(dev), log_density_t[idx].to(dev)
        for b in range(batch_size):
            mb[b, 0], db[b, 0] = augment(mb[b, 0], db[b, 0])

        pred = model(mb)
        loss_map = (pred - db) ** 2 * mb  # only inside the boundary mask
        loss = loss_map.sum() / mb.sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append((step, loss.item()))
        if step % 200 == 0 or step == steps - 1:
            print(f"step {step:4d} loss={loss.item():.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "d6_density_model.pt")
    np.save(OUT_DIR / "d6_train_history.npy", np.array(history))
    np.savez(OUT_DIR / "d6_config.npz", base_ch=base_ch, grid_res=GRID_RES)
    print(f"saved model + history to {OUT_DIR}")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    train(steps=steps)
