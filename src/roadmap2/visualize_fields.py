"""Sanity-check the extracted control fields (ROADMAP2's stated validation
step, before trusting them as R2 training targets) -- quiver plot for
alignment flow, scatter/heatmap for the size gradient, on a handful of real
images across all three datasets.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.roadmap2.control_fields import add_control_fields

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "roadmap2"


def visualize_one(image_name: str, tag: str = "norm"):
    graph_dir = out_dir_for(tag)
    cached = torch.load(graph_dir / f"{image_name}.pt", weights_only=False)
    node_df = add_control_fields(cached["node_df"])

    row, col = node_df["centroid_row"].values, node_df["centroid_col"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(col, -row, c=node_df["size_gradient_target"], cmap="viridis", s=8)
    axes[0].set_title(f"{image_name}: size_gradient_target\n(leave-one-out local mean equivalent_diameter_norm)")
    axes[0].set_aspect("equal")
    fig.colorbar(sc, ax=axes[0])

    # Quiver: axial (180-deg-periodic) orientation -- draw as a double-headed
    # arrow by plotting +theta and -theta together (no single arrowhead makes
    # sense for an axial direction).
    step = max(1, len(node_df) // 400)  # subsample for a readable quiver plot
    r = row[::step]
    c = col[::step]
    ang = node_df["alignment_flow_angle"].values[::step]
    coh = node_df["alignment_flow_coherence"].values[::step]
    u, v = np.cos(ang), np.sin(ang)
    axes[1].quiver(c, -r, u, -v, coh, cmap="plasma", scale=40, width=0.003, pivot="middle")
    axes[1].quiver(c, -r, -u, v, coh, cmap="plasma", scale=40, width=0.003, pivot="middle")
    axes[1].set_title(f"{image_name}: alignment_flow_angle (color = coherence)")
    axes[1].set_aspect("equal")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"control_fields_{image_name}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}  (n_cells={len(node_df)}, "
          f"mean coherence={node_df['alignment_flow_coherence'].mean():.3f}, "
          f"size_gradient_target range=[{node_df['size_gradient_target'].min():.3f}, "
          f"{node_df['size_gradient_target'].max():.3f}])")


if __name__ == "__main__":
    images = sys.argv[1:] or ["DO_0000", "EH_0049", "VM_0033"]
    for name in images:
        visualize_one(name)
