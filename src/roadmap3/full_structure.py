"""Generate a full structure -- all of DO_0000's real cells (6,748) in a
single DDIM pass, real positions + DO_0000's own real control fields (not
tiled, not synthetic). This is the actual "whole-structure generation, cost
at scale" question ROADMAP2 R4 and ROADMAP3's D4/D5 open-questions section
both flagged as unmeasured -- full self-attention across the whole cell set
in one connected pass is exactly what this roadmap's motivation (ROADMAP2's
"no seams by construction") requires, unlike Phase5/6's independently-
generated, stitched pixel tiles.

geometry.py's power-diagram renderer is pure-Python and O(N^2) per site --
fine at patch scale (n~100-200), far too slow at n=6748 (~3000x more
site-pairs than the 124-cell DO_0000 subsection test). Generation itself is
cheap (attention is a single batched op, ~0.37s/forward pass measured
directly), so this only renders a small spatial CROP of the full generated
structure for visual inspection, while computing correlation/distribution
statistics over the FULL 6,748-cell generated set (cheap, pure numpy).
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.roadmap2.control_fields import add_control_fields
from src.roadmap2.train_infill import CTRL_COLS
from src.roadmap3.geometry import cell_positions_and_weights, is_boundary_cell, power_diagram
from src.roadmap3.render import polygon_area
from src.roadmap3.sample_poscond import FEATURE_NAMES, POS_IDX, generate, load_model
from src.roadmap3.train import OUT_DIR


def load_do0000_full():
    cached = torch.load(out_dir_for("norm") / "DO_0000.pt", weights_only=False)
    node_df = add_control_fields(cached["node_df"])
    node_df["_flow_sin"] = np.sin(node_df["alignment_flow_angle"])
    node_df["_flow_cos"] = np.cos(node_df["alignment_flow_angle"])
    ctrl = torch.tensor(node_df[CTRL_COLS].values, dtype=torch.float32)
    return cached["data"].x, ctrl


def run(num_inference_steps: int = 50, seed: int = 0, crop_frac: float = 0.12):
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    x_raw, ctrl_raw = load_do0000_full()
    n = x_raw.size(0)
    print(f"DO_0000 full structure: {n} real cells")

    pos_raw = x_raw[:, POS_IDX]
    t0 = time.time()
    x_gen = generate(model, pos_raw, ctrl_raw, mean, std, ctrl_mean, ctrl_std, dev,
                      num_train_timesteps, num_inference_steps, seed=seed)
    gen_time = time.time() - t0
    print(f"generation time (all {n} cells, one connected DDIM pass, {num_inference_steps} steps): "
          f"{gen_time:.1f}s")

    diam_idx = FEATURE_NAMES.index("equivalent_diameter_norm")
    corr = float(np.corrcoef(ctrl_raw[:, 0].numpy(), x_gen[:, diam_idx])[0, 1])
    print(f"whole-structure control-following: corr(size_gradient_target, generated diameter) = {corr:.3f}"
          f"  (n={n}, far more stable than any patch-scale estimate so far)")

    x_real = x_raw.numpy()
    print(f"\n{'feature':<24} {'real mean':>12} {'gen mean':>12} {'real std':>12} {'effect size':>12}")
    for name in ["area_norm", "elongation", "orientation"]:
        idx = FEATURE_NAMES.index(name)
        r_mean, r_std = x_real[:, idx].mean(), x_real[:, idx].std()
        g_mean = x_gen[:, idx].mean()
        effect = abs(g_mean - r_mean) / max(r_std, 1e-8)
        print(f"{name:<24} {r_mean:>12.4g} {g_mean:>12.4g} {r_std:>12.4g} {effect:>12.3f}")

    # Crop a small spatial window (fraction of the tissue's radial extent)
    # for the only part slow enough to need it -- geometry rendering.
    r = pos_raw[:, 0].numpy()
    center_r = np.median(r)
    r_max = r.max()
    window = crop_frac * r_max
    theta = pos_raw[:, 1].numpy()
    xy_all = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    center_xy = np.median(xy_all, axis=0)
    dist = np.linalg.norm(xy_all - center_xy, axis=1)
    crop_mask = dist < window
    print(f"\nrendering a {crop_mask.sum()}-cell crop (of {n}) for visual inspection")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, attrs, title in [(axes[0], x_real[crop_mask], "real DO_0000 (crop)"),
                              (axes[1], x_gen[crop_mask], "generated (full-structure DDIM, crop shown)")]:
        points, weights = cell_positions_and_weights(attrs, FEATURE_NAMES)
        cells, box = power_diagram(points, weights)
        boundary = np.array([is_boundary_cell(c, box) for c in cells])
        areas = np.array([polygon_area(c) if len(c) >= 3 else np.nan for c in cells])
        drawable = [(c, a) for c, a, b in zip(cells, areas, boundary) if len(c) >= 3 and not b]
        if drawable:
            pc = PolyCollection([c for c, _ in drawable], array=np.array([a for _, a in drawable]),
                                 cmap="viridis", edgecolors="white", linewidths=0.3)
            ax.add_collection(pc)
        ax.scatter(points[:, 0], points[:, 1], s=2, color="red", zorder=3)
        m = 0.1 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1e-6)
        ax.set_xlim(points[:, 0].min() - m, points[:, 0].max() + m)
        ax.set_ylim(points[:, 1].min() - m, points[:, 1].max() + m)
        ax.set_aspect("equal")
        ax.set_title(title)

    fig.suptitle(f"DO_0000 whole-structure generation (n={n} cells, {gen_time:.1f}s, "
                 f"whole-structure corr={corr:.3f}) -- crop shown for rendering cost only")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "do0000_full_structure.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return corr, gen_time


if __name__ == "__main__":
    run()
