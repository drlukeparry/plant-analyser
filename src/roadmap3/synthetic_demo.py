"""D5 (first pass): infill a synthetic boundary with an explicit, novel
control field -- not lifted from any real image. This is the actual
generalization test ROADMAP2/ROADMAP3 both call for: everything upstream
(D1-D3.5) only ever regenerated within a real subgraph's own footprint,
conditioned on that image's own extracted control values.

Circle boundary, radius 0.15 (matching the spatial scale of the real k-hop
patches the model was trained on -- this is a patch-scale test, not the
larger whole-structure-generation ambition D5 also describes; see
ROADMAP3.md's own scope note). Size gradient is deliberately steeper than
typical real images (diam_center=0.010 vs. diam_edge=0.003, real dataset
mean+/-std was 0.0059+/-0.0042) -- per D5's own suggestion that gradients
"not seen in training" are the actual point, not a gentle in-distribution
one.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.roadmap3.geometry import cell_positions_and_weights, is_boundary_cell, power_diagram
from src.roadmap3.render import polygon_area
from src.roadmap3.sample_poscond import FEATURE_NAMES, generate, load_model
from src.roadmap3.synthetic import (circular_boundary, radial_size_field,
                                     seed_positions_blue_noise, tangential_flow_field)
from src.roadmap3.train import OUT_DIR

POS_IDX = [FEATURE_NAMES.index("radial_distance_norm"), FEATURE_NAMES.index("angular_position")]
DIAM_IDX = FEATURE_NAMES.index("equivalent_diameter_norm")


def build_ctrl_and_positions(center: np.ndarray, radius: float, diam_center: float, diam_edge: float,
                              seed: int = 0, boundary_fn=None):
    inside = boundary_fn if boundary_fn is not None else circular_boundary(center, radius)
    size_fn = radial_size_field(center, radius, diam_center, diam_edge)
    flow_fn = tangential_flow_field(center)
    bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    xy = seed_positions_blue_noise(inside, size_fn, bbox, seed=seed)

    n = len(xy)
    ctrl_raw = np.zeros((n, 4), dtype=np.float32)  # CTRL_COLS order: size_gradient_target, _flow_sin, _flow_cos, coherence
    for i, p in enumerate(xy):
        ctrl_raw[i, 0] = size_fn(p)
        angle, coherence = flow_fn(p)
        ctrl_raw[i, 1] = np.sin(angle)
        ctrl_raw[i, 2] = np.cos(angle)
        ctrl_raw[i, 3] = coherence

    r = np.linalg.norm(xy - center, axis=1)
    theta = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
    pos_raw = np.stack([r, theta], axis=1).astype(np.float32)
    return xy, pos_raw, ctrl_raw, size_fn


def run(radius: float = 0.15, diam_center: float = 0.010, diam_edge: float = 0.003,
        num_inference_steps: int = 50, seed: int = 0, boundary_fn=None, boundary_patch=None,
        shape_name: str = "circle", out_name: str = "d5_synthetic_infill_poscond.png"):
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    center = np.array([0.0, 0.0])

    xy, pos_raw, ctrl_raw, size_fn = build_ctrl_and_positions(center, radius, diam_center, diam_edge, seed,
                                                                boundary_fn=boundary_fn)
    print(f"seeded {len(xy)} cell positions inside the synthetic boundary")

    pos_raw_t = torch.tensor(pos_raw, dtype=torch.float32)
    ctrl_raw_t = torch.tensor(ctrl_raw, dtype=torch.float32)
    x_gen = generate(model, pos_raw_t, ctrl_raw_t, mean, std, ctrl_mean, ctrl_std, dev,
                      num_train_timesteps, num_inference_steps, seed=seed)

    requested_diam = ctrl_raw[:, 0]
    generated_diam = x_gen[:, DIAM_IDX]
    corr = float(np.corrcoef(requested_diam, generated_diam)[0, 1])
    print(f"generalization check: corr(requested synthetic size field, generated diameter) = {corr:.3f}")

    points, weights = cell_positions_and_weights(x_gen, FEATURE_NAMES)
    cells, box = power_diagram(points, weights)
    boundary_cell = np.array([is_boundary_cell(c, box) for c in cells])
    areas = np.array([polygon_area(c) if len(c) >= 3 else np.nan for c in cells])

    def make_patch():
        if boundary_patch is not None:
            return boundary_patch()
        return plt.Circle(center, radius, fill=False, color="black", linestyle="--")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=requested_diam, cmap="viridis", s=18)
    ax.add_patch(make_patch())
    ax.set_aspect("equal")
    ax.set_title(f"requested size field (input, {shape_name} boundary)")
    fig.colorbar(sc, ax=ax, label="requested equivalent_diameter_norm")

    ax = axes[1]
    from matplotlib.collections import PolyCollection
    drawable = [(c, a) for c, a, b in zip(cells, areas, boundary_cell) if len(c) >= 3 and not b]
    if drawable:
        pc = PolyCollection([c for c, _ in drawable], array=np.array([a for _, a in drawable]),
                             cmap="viridis", edgecolors="white", linewidths=0.4)
        ax.add_collection(pc)
    ax.scatter(points[:, 0], points[:, 1], s=3, color="red", zorder=3)
    ax.add_patch(make_patch())
    m = 0.15 * radius
    ax.set_xlim(center[0] - radius - m, center[0] + radius + m)
    ax.set_ylim(center[1] - radius - m, center[1] + radius + m)
    ax.set_aspect("equal")
    ax.set_title(f"generated infill, corr(requested, generated size)={corr:.3f}")

    fig.suptitle(f"D5: synthetic {shape_name} boundary + novel size gradient/flow field, not seen in training")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return corr, len(xy)


if __name__ == "__main__":
    run()
