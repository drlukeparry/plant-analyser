"""Attempt to replicate a real subsection of DO_0000 (the reference
specimen used throughout PLAN.md/Phase5/6) with the position-conditioned
set-Transformer DDIM: take a real k-hop patch of DO_0000's own cells, keep
its real positions and its own real extracted control fields (not pooled/
leave-one-out across other images -- DO_0000's own local size-gradient and
alignment-flow trend), and regenerate everything else (size, shape,
orientation) from scratch via DDIM, conditioned on that real
position+control input. This is the bounded, real-image test recommended
after D5's generalization-gap finding: same real boundary/control-field
scale D3 already validated (0.364 pooled control-following, no synthetic
seeding involved), but now specifically on DO_0000 and at a larger patch
size (up to 200 cells, vs. D3's 64) for a more substantial "subsection"
demonstration.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.roadmap2.control_fields import add_control_fields
from src.roadmap3.geometry import cell_positions_and_weights, is_boundary_cell, power_diagram
from src.roadmap3.render import polygon_area
from src.roadmap3.sample_poscond import ATTR_IDX, FEATURE_NAMES, POS_IDX, generate, load_model
from src.roadmap2.train_infill import CTRL_COLS
from src.roadmap3.train import OUT_DIR


def load_do0000_subsection(seed_node: int | None = None, num_hops: int = 6, max_nodes: int = 200,
                            seed: int = 0):
    cached = torch.load(out_dir_for("norm") / "DO_0000.pt", weights_only=False)
    node_df = add_control_fields(cached["node_df"])
    node_df["_flow_sin"] = np.sin(node_df["alignment_flow_angle"])
    node_df["_flow_cos"] = np.cos(node_df["alignment_flow_angle"])
    ctrl = torch.tensor(node_df[CTRL_COLS].values, dtype=torch.float32)
    data = cached["data"]

    torch.manual_seed(seed)
    if seed_node is None:
        seed_node = torch.randint(0, data.num_nodes, (1,)).item()
    node_idx, _, _, _ = k_hop_subgraph(seed_node, num_hops, data.edge_index, relabel_nodes=True,
                                        num_nodes=data.num_nodes)
    if node_idx.numel() > max_nodes:
        keep = torch.randperm(node_idx.numel())[:max_nodes]
        node_idx = node_idx[keep]
    return data.x[node_idx], ctrl[node_idx], seed_node


def run(seed_node: int | None = 3000, num_hops: int = 6, max_nodes: int = 200, seed: int = 0):
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    x_raw, ctrl_raw, seed_node = load_do0000_subsection(seed_node, num_hops, max_nodes, seed)
    n = x_raw.size(0)
    print(f"DO_0000 subsection: seed_node={seed_node}, {n} cells (num_hops={num_hops}, max_nodes={max_nodes})")

    pos_raw = x_raw[:, POS_IDX]
    x_gen = generate(model, pos_raw, ctrl_raw, mean, std, ctrl_mean, ctrl_std, dev,
                      num_train_timesteps, seed=seed)

    diam_idx = FEATURE_NAMES.index("equivalent_diameter_norm")
    corr = float(np.corrcoef(ctrl_raw[:, 0].numpy(), x_gen[:, diam_idx])[0, 1])
    print(f"control-following on this subsection: corr(size_gradient_target, generated diameter) = {corr:.3f}")

    for name in ["area_norm", "elongation", "orientation"]:
        idx = FEATURE_NAMES.index(name)
        r_mean, r_std = x_raw[:, idx].mean().item(), x_raw[:, idx].std().item()
        g_mean = x_gen[:, idx].mean()
        effect = abs(g_mean - r_mean) / max(r_std, 1e-8)
        print(f"  {name:<24} real={r_mean:.4g} gen={g_mean:.4g} effect_size={effect:.3f}")

    x_real = x_raw.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, attrs, title in [(axes[0], x_real, "real DO_0000 subsection"),
                              (axes[1], x_gen, "regenerated (real positions + real control fields)")]:
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

    fig.suptitle(f"DO_0000 subsection replication (n={n} cells), "
                 f"corr(requested size, generated size)={corr:.3f}")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "do0000_replication.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return corr


if __name__ == "__main__":
    run()
