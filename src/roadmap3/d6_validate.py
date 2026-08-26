"""D6 prototype validation, two parts:
1. Visual: predicted vs. real density raster on a few real images (same
   in-sample caveat as every other checkpoint in this roadmap -- no
   held-out image split exists yet).
2. The new validation axis D6's plan calls for and this roadmap hasn't
   needed before: does POINT PLACEMENT alone look right, independent of
   attribute quality? Sample points from the predicted density field via
   density-weighted blue noise (reusing synthetic.py's dart-throwing,
   spacing derived from the model's own predicted density -- not a
   hand-specified field like D5's), then compare the generated point set's
   nearest-neighbor distance distribution against the real point set's own
   -- this is the placement-quality check ROADMAP3.md's D6 section
   specifically flagged as new and not covered by any prior D3/D5 check.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.models.gvae.train import device
from src.roadmap2.control_fields import _normalized_xy
from src.roadmap3.d6_density_field import (GRID_EXTENT, GRID_RES, density_at,
                                            density_to_spacing_diam, rasterize_image)
from src.roadmap3.d6_model import DensityFieldNet
from src.roadmap3.synthetic import seed_positions_blue_noise
from src.roadmap3.train import OUT_DIR


def load_model():
    dev = device()
    config = np.load(OUT_DIR / "d6_config.npz")
    model = DensityFieldNet(base_ch=int(config["base_ch"])).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / "d6_density_model.pt", map_location=dev))
    model.eval()
    return model, dev


@torch.no_grad()
def predict_density(model, boundary_mask: np.ndarray, dev) -> np.ndarray:
    x = torch.tensor(boundary_mask, dtype=torch.float32).view(1, 1, *boundary_mask.shape).to(dev)
    pred_log = model(x).squeeze().cpu().numpy()
    return np.expm1(np.clip(pred_log, None, 20)) * boundary_mask  # invert log1p, re-zero outside mask


def nn_distance_distribution(xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)
    return d[:, 1]


def visual_check(image_names=("DO_0000", "EH_0049", "VM_0033")):
    model, dev = load_model()
    fig, axes = plt.subplots(2, len(image_names), figsize=(4.3 * len(image_names), 8))
    for col, name in enumerate(image_names):
        cached = torch.load(out_dir_for("norm") / f"{name}.pt", weights_only=False)
        mask, real_density = rasterize_image(cached["node_df"])
        pred_density = predict_density(model, mask, dev)

        vmax = real_density.max()
        axes[0, col].imshow(real_density, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[0, col].set_title(f"{name} real density")
        axes[1, col].imshow(pred_density, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axes[1, col].set_title(f"{name} predicted density")
    fig.tight_layout()
    out = OUT_DIR / "d6_density_prediction_check.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def placement_check(image_names=("DO_0000", "EH_0049", "VM_0033"), seed: int = 0):
    model, dev = load_model()
    fig, axes = plt.subplots(1, len(image_names), figsize=(4.5 * len(image_names), 4.2))
    for col, name in enumerate(image_names):
        cached = torch.load(out_dir_for("norm") / f"{name}.pt", weights_only=False)
        node_df = cached["node_df"]
        real_xy = _normalized_xy(node_df)
        mask, _ = rasterize_image(node_df)
        pred_density = predict_density(model, mask, dev)

        def inside(p, mask=mask):
            cell = (2 * GRID_EXTENT) / (GRID_RES - 1)
            ix = int(np.clip(round((p[0] + GRID_EXTENT) / cell), 0, GRID_RES - 1))
            iy = int(np.clip(round((p[1] + GRID_EXTENT) / cell), 0, GRID_RES - 1))
            return mask[iy, ix] > 0.5

        def size_fn(p, pred_density=pred_density):
            d = density_at(pred_density, np.array([p]))[0]
            return density_to_spacing_diam(np.array([d]))[0]

        bbox = (real_xy[:, 0].min(), real_xy[:, 1].min(), real_xy[:, 0].max(), real_xy[:, 1].max())
        gen_xy = seed_positions_blue_noise(inside, size_fn, bbox, spacing_factor=1.0,
                                            max_points=len(real_xy), seed=seed)

        real_nn = nn_distance_distribution(real_xy)
        gen_nn = nn_distance_distribution(gen_xy) if len(gen_xy) > 2 else np.array([])

        ax = axes[col]
        ax.hist(real_nn, bins=30, alpha=0.5, density=True, label=f"real (n={len(real_xy)})")
        if len(gen_nn) > 0:
            ax.hist(gen_nn, bins=30, alpha=0.5, density=True, label=f"generated (n={len(gen_xy)})")
        ax.set_title(name)
        ax.set_xlabel("nearest-neighbor distance")
        ax.legend(fontsize=8)
        print(f"{name}: real NN dist mean/std = {real_nn.mean():.4f}/{real_nn.std():.4f}, "
              f"n={len(real_xy)}  |  generated n={len(gen_xy)}, "
              f"mean/std = {gen_nn.mean() if len(gen_nn) else float('nan'):.4f}/"
              f"{gen_nn.std() if len(gen_nn) else float('nan'):.4f}")

    fig.suptitle("D6 placement check: nearest-neighbor distance distribution, real vs. generated points")
    fig.tight_layout()
    out = OUT_DIR / "d6_placement_check.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def placement_check_real_density(image_names=("DO_0000", "EH_0049", "VM_0033"), seed: int = 0):
    """Same as placement_check, but samples from the REAL density raster
    instead of the model's prediction -- isolates Finding 2 (does the
    dart-throwing sampler under-fill even given a perfect density field?)
    from Finding 1 (does the model predict the right density level?). If
    under-filling persists here, it's purely a sampler problem; if it
    mostly resolves, Finding 2 was actually downstream of Finding 1."""
    fig, axes = plt.subplots(1, len(image_names), figsize=(4.5 * len(image_names), 4.2))
    for col, name in enumerate(image_names):
        cached = torch.load(out_dir_for("norm") / f"{name}.pt", weights_only=False)
        node_df = cached["node_df"]
        real_xy = _normalized_xy(node_df)
        mask, real_density = rasterize_image(node_df)

        def inside(p, mask=mask):
            cell = (2 * GRID_EXTENT) / (GRID_RES - 1)
            ix = int(np.clip(round((p[0] + GRID_EXTENT) / cell), 0, GRID_RES - 1))
            iy = int(np.clip(round((p[1] + GRID_EXTENT) / cell), 0, GRID_RES - 1))
            return mask[iy, ix] > 0.5

        def size_fn(p, real_density=real_density):
            d = density_at(real_density, np.array([p]))[0]
            return density_to_spacing_diam(np.array([d]))[0]

        bbox = (real_xy[:, 0].min(), real_xy[:, 1].min(), real_xy[:, 0].max(), real_xy[:, 1].max())
        gen_xy = seed_positions_blue_noise(inside, size_fn, bbox, spacing_factor=1.0,
                                            max_points=len(real_xy), seed=seed)

        real_nn = nn_distance_distribution(real_xy)
        gen_nn = nn_distance_distribution(gen_xy) if len(gen_xy) > 2 else np.array([])

        ax = axes[col]
        ax.hist(real_nn, bins=30, alpha=0.5, density=True, label=f"real (n={len(real_xy)})")
        if len(gen_nn) > 0:
            ax.hist(gen_nn, bins=30, alpha=0.5, density=True, label=f"generated (n={len(gen_xy)})")
        ax.set_title(name)
        ax.set_xlabel("nearest-neighbor distance")
        ax.legend(fontsize=8)
        print(f"{name}: real NN dist mean/std = {real_nn.mean():.4f}/{real_nn.std():.4f}, "
              f"n={len(real_xy)}  |  generated (real density) n={len(gen_xy)}, "
              f"fill={100*len(gen_xy)/len(real_xy):.1f}%, "
              f"mean/std = {gen_nn.mean() if len(gen_nn) else float('nan'):.4f}/"
              f"{gen_nn.std() if len(gen_nn) else float('nan'):.4f}")

    fig.suptitle("D6 placement check (REAL density field): nearest-neighbor distance, real vs. generated points")
    fig.tight_layout()
    out = OUT_DIR / "d6_placement_check_real_density.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    visual_check()
    placement_check()
    placement_check_real_density()
