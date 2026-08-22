"""Phase 3 evaluation: does the GVAE latent space separate by tissue
composition (radial position) without being told to? Per PLAN.md exit
criteria: cluster the latent space, check clusters track visible tissue rings.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_graph import NODE_FEATURE_COLS_PX, build_graph
from src.models.gvae.model import GraphVAE
from src.models.gvae.sampling import sample_subgraph
from src.models.gvae.train import device

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"


def pca_2d(x: np.ndarray) -> np.ndarray:
    """No sklearn dependency — plain SVD-based PCA."""
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def evaluate(labels_path: str, n_samples: int = 800, num_hops: int = 3, max_nodes: int = 300):
    dev = device()
    labels = np.load(labels_path)
    full_data, node_df = build_graph(labels)

    scaler = np.load(OUT_DIR / "feature_scaler.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    full_data_std = full_data.clone()
    full_data_std.x = (full_data.x - mean) / std

    model = GraphVAE(in_dim=full_data.x.size(1)).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / "gvae.pt", map_location=dev))
    model.eval()

    zs, mean_radial, mean_elong, mean_area = [], [], [], []
    radial_col = NODE_FEATURE_COLS_PX.index("radial_distance")
    elong_col = NODE_FEATURE_COLS_PX.index("elongation")
    area_col = NODE_FEATURE_COLS_PX.index("area")

    with torch.no_grad():
        for _ in range(n_samples):
            sub = sample_subgraph(full_data_std, num_hops=num_hops, max_nodes=max_nodes)
            if sub.num_nodes < 8:
                continue
            batch = torch.zeros(sub.num_nodes, dtype=torch.long)
            mu, _ = model.encode(sub.x.to(dev), sub.edge_index.to(dev), batch.to(dev))
            zs.append(mu.cpu().numpy()[0])

            # unstandardize this subgraph's features to get interpretable stats
            sub_x_raw = sub.x * std + mean
            mean_radial.append(sub_x_raw[:, radial_col].mean().item())
            mean_elong.append(sub_x_raw[:, elong_col].mean().item())
            mean_area.append(sub_x_raw[:, area_col].mean().item())

    zs = np.array(zs)
    mean_radial = np.array(mean_radial)
    mean_elong = np.array(mean_elong)
    mean_area = np.array(mean_area)
    print(f"collected {len(zs)} subgraph latents, z dim={zs.shape[1]}")

    proj = pca_2d(zs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, color_by, name in zip(axes, [mean_radial, mean_elong, mean_area], ["mean radial_distance", "mean elongation", "mean area"]):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_by, cmap="viridis", s=8, alpha=0.7)
        ax.set_title(f"latent space (PCA) colored by {name}")
        fig.colorbar(sc, ax=ax, fraction=0.03)
    fig.tight_layout()
    out_path = OUT_DIR / "latent_space_pca.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    # crude quantitative check: correlation between latent PC1 and radial_distance
    corr = np.corrcoef(proj[:, 0], mean_radial)[0, 1]
    print(f"corr(latent PC1, mean radial_distance) = {corr:.3f}")


if __name__ == "__main__":
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase1/DO_0000_full_labels.npy"
    evaluate(labels_path)
