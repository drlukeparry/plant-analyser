"""Wires geometry.py's power-diagram renderer to the trained set-Transformer
DDIM: sample a real subgraph's control fields, generate a cell set with it
(same DDIM sampling as sample.py), render both the real and generated cell
sets as tessellations side by side, and check that rendered polygon area
actually tracks each cell's own generated size (the geometry step
introduces its own potential failure mode -- e.g. degenerate/absorbed
cells at the tessellation's edges -- on top of whatever the denoiser itself
gets right or wrong, so this is checked separately rather than assumed).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.roadmap3.dataset import load_graphs_with_ctrl, sample_token_subgraph
from src.roadmap3.geometry import cell_positions_and_weights, is_boundary_cell, power_diagram
from src.roadmap3.sample import FEATURE_NAMES, generate, load_model
from src.roadmap3.train import OUT_DIR


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def render_cells(ax, attrs: np.ndarray, title: str):
    """Returns (areas, interior_mask) -- boundary cells (touching the
    power diagram's arbitrary clip box, see geometry.is_boundary_cell) get
    NaN area and interior_mask=False, since their extent is a clip-box
    artifact, not a function of the cell's own generated/real size."""
    points, weights = cell_positions_and_weights(attrs, FEATURE_NAMES)
    cells, box = power_diagram(points, weights)
    boundary = np.array([is_boundary_cell(c, box) for c in cells])
    areas = np.array([polygon_area(c) if len(c) >= 3 else np.nan for c in cells])
    areas_for_color = np.where(boundary, np.nan, areas)

    drawable = [(c, a) for c, a, b in zip(cells, areas_for_color, boundary) if len(c) >= 3]
    pc = PolyCollection([c for c, _ in drawable], array=np.array([a for _, a in drawable]),
                         cmap="viridis", edgecolors="white", linewidths=0.4)
    ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], s=3, color="red", zorder=3)

    m = 0.15 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1e-6)
    ax.set_xlim(points[:, 0].min() - m, points[:, 0].max() + m)
    ax.set_ylim(points[:, 1].min() - m, points[:, 1].max() + m)
    ax.set_aspect("equal")
    n_interior = int((~boundary & (areas > 0)).sum())
    ax.set_title(f"{title}\n{n_interior}/{len(cells)} interior cells (rest touch clip boundary)")
    return areas, ~boundary


def render_comparison(n_examples: int = 5, seed: int = 2):
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    graphs, ctrls = load_graphs_with_ctrl("norm")

    torch.manual_seed(seed)
    fig, axes = plt.subplots(2, n_examples, figsize=(5 * n_examples, 10))
    diam_idx = FEATURE_NAMES.index("equivalent_diameter_norm")
    corrs = []

    for col in range(n_examples):
        gi = torch.randint(0, len(graphs), (1,)).item()
        x, c = sample_token_subgraph(graphs[gi], ctrls[gi], num_hops=3, max_nodes=64)
        while x.size(0) < 12:
            gi = torch.randint(0, len(graphs), (1,)).item()
            x, c = sample_token_subgraph(graphs[gi], ctrls[gi], num_hops=3, max_nodes=64)
        x_real = (x * std + mean).numpy()
        x_gen = generate(model, c, mean, std, ctrl_mean, ctrl_std, dev, num_train_timesteps,
                          seed=seed * 1000 + col).numpy()

        real_areas, real_interior = render_cells(axes[0, col], x_real, f"real (n={len(x_real)})")
        gen_areas, gen_interior = render_cells(axes[1, col], x_gen, f"generated (n={len(x_gen)})")

        real_target = np.pi * (x_real[:, diam_idx] / 2) ** 2
        gen_target = np.pi * (x_gen[:, diam_idx] / 2) ** 2  # implied disc area from own generated/real size
        real_mask = real_interior & (real_areas > 0)
        gen_mask = gen_interior & (gen_areas > 0)
        if real_mask.sum() > 3 and gen_mask.sum() > 3:
            corrs.append((
                float(np.corrcoef(real_target[real_mask], real_areas[real_mask])[0, 1]),
                float(np.corrcoef(gen_target[gen_mask], gen_areas[gen_mask])[0, 1]),
            ))

    axes[0, 0].set_ylabel("real cells")
    axes[1, 0].set_ylabel("generated cells")
    real_corrs = [c[0] for c in corrs]
    gen_corrs = [c[1] for c in corrs]
    mean_real, mean_gen = float(np.mean(real_corrs)), float(np.mean(gen_corrs))
    fig.suptitle(
        "R3 geometry render: power-diagram tessellation from generated centroids + sizes\n"
        f"corr(own diameter, rendered polygon area): real={mean_real:.3f}  generated={mean_gen:.3f}"
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "d3_geometry_render.png"
    fig.savefig(out_path, dpi=130)
    print(f"per-example (real, generated) correlations: {corrs}")
    print(f"mean real={mean_real:.3f}  mean generated={mean_gen:.3f}")
    print(f"saved {out_path}")
    return mean_real, mean_gen


if __name__ == "__main__":
    render_comparison()
