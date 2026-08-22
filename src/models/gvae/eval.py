"""Phase 3 evaluation: does the GVAE latent space separate by tissue
composition (radial position) without being told to? Per PLAN.md exit
criteria: cluster the latent space, check clusters track visible tissue rings.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.build_graph import NODE_FEATURE_COLS_PX, build_graph
from src.models.gvae.model import GraphVAE
from src.models.gvae.sampling import sample_subgraph
from src.models.gvae.train import device

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"

# Confirmed against literature/2212.03022v2.pdf (Ganz et al., INBD) Table 1 --
# image counts match exactly (DO 22+42=64, EH 24+58=82, VM 22+45=67).
DATASET_SPECIES = {
    "DO": "Dryas octopetala",
    "EH": "Empetrum hermaphroditum",
    "VM": "Vaccinium myrtillus",
}


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Plain SVD-based PCA -- linear, so its axes stay directly
    interpretable for the corr()/variance-fraction checks below."""
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def tsne_2d(x: np.ndarray, perplexity: float = 30.0, seed: int = 0) -> np.ndarray:
    """t-SNE projection -- nonlinear, better at revealing cluster structure
    (e.g. does the latent space actually separate into distinct
    per-species blobs) than PCA's linear projection can show, at the cost
    of axes/distances that aren't otherwise meaningful. Used here purely
    for visual cluster inspection, not for the quantitative variance-ratio
    checks (those stay on PCA's linear axis)."""
    perplexity = min(perplexity, (len(x) - 1) / 3)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(x)


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


def evaluate_multi(graph_paths: list[Path], n_samples: int = 1600, num_hops: int = 3,
                    max_nodes: int = 300, tag: str = "multi"):
    """Same evaluation as `evaluate`, but samples subgraphs across every
    cached image graph (not just one), using the multi-image checkpoint +
    feature scaler from train_multi. Each drawn subgraph is tagged by its
    source image so the PCA plot can also be colored by source, to check
    whether the latent space clusters by image identity (a red flag -- it
    should cluster by tissue composition, not by which specimen a patch came
    from) rather than only by radial position.
    """
    dev = device()
    print(f"loading {len(graph_paths)} cached graphs ...")
    names, graphs = [], []
    for p in graph_paths:
        cached = torch.load(p, weights_only=False)
        names.append(p.stem)
        graphs.append(cached["data"])

    scaler = np.load(OUT_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    graphs_std = []
    for g in graphs:
        gs = g.clone()
        gs.x = (g.x - mean) / std
        graphs_std.append(gs)

    model = GraphVAE(in_dim=graphs[0].x.size(1)).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / f"gvae_{tag}.pt", map_location=dev))
    model.eval()

    # dataset/species label per graph, parsed from the cached filename prefix
    # (e.g. "DO_0022" -> "DO") -- see DATASET_SPECIES for the confirmed
    # species behind each code.
    dataset_of_graph = [n.split("_")[0] for n in names]
    species_codes = sorted(set(dataset_of_graph))

    zs, mean_radial, mean_elong, mean_area, source_idx, dataset_idx = [], [], [], [], [], []
    radial_col = NODE_FEATURE_COLS_PX.index("radial_distance")
    elong_col = NODE_FEATURE_COLS_PX.index("elongation")
    area_col = NODE_FEATURE_COLS_PX.index("area")

    with torch.no_grad():
        for _ in range(n_samples):
            gi = np.random.randint(len(graphs_std))
            sub = sample_subgraph(graphs_std[gi], num_hops=num_hops, max_nodes=max_nodes)
            if sub.num_nodes < 8:
                continue
            batch = torch.zeros(sub.num_nodes, dtype=torch.long)
            mu, _ = model.encode(sub.x.to(dev), sub.edge_index.to(dev), batch.to(dev))
            zs.append(mu.cpu().numpy()[0])

            sub_x_raw = sub.x * std + mean
            mean_radial.append(sub_x_raw[:, radial_col].mean().item())
            mean_elong.append(sub_x_raw[:, elong_col].mean().item())
            mean_area.append(sub_x_raw[:, area_col].mean().item())
            source_idx.append(gi)
            dataset_idx.append(species_codes.index(dataset_of_graph[gi]))

    zs = np.array(zs)
    mean_radial = np.array(mean_radial)
    mean_elong = np.array(mean_elong)
    mean_area = np.array(mean_area)
    source_idx = np.array(source_idx)
    dataset_idx = np.array(dataset_idx)
    print(f"collected {len(zs)} subgraph latents from {len(graphs)} images, z dim={zs.shape[1]}")

    def plot_panels(proj, embedding_name, out_path):
        fig, axes = plt.subplots(1, 5, figsize=(30, 5.5))
        for ax, color_by, name, cmap in zip(
            axes[:4],
            [mean_radial, mean_elong, mean_area, source_idx],
            ["mean radial_distance", "mean elongation", "mean area", "source image (index)"],
            ["viridis", "viridis", "viridis", "tab20"],
        ):
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_by, cmap=cmap, s=8, alpha=0.7)
            ax.set_title(f"latent space ({embedding_name}) colored by {name}")
            fig.colorbar(sc, ax=ax, fraction=0.03)

        # dataset/species panel: discrete legend, not a colorbar, since there
        # are only len(species_codes) categories -- this is the panel that
        # visually confirms (or refutes) a genuine species-level separation.
        ax = axes[4]
        palette = plt.get_cmap("tab10").colors
        for i, code in enumerate(species_codes):
            m = dataset_idx == i
            label = f"{code} ({DATASET_SPECIES.get(code, 'unknown')})"
            ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.7, color=palette[i % 10], label=label)
        ax.set_title(f"latent space ({embedding_name}) colored by species")
        ax.legend(fontsize=7, markerscale=2, loc="best")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"saved {out_path}")

    proj = pca_2d(zs)
    plot_panels(proj, "PCA", OUT_DIR / f"latent_space_pca_{tag}.png")

    print("running t-SNE ...")
    proj_tsne = tsne_2d(zs)
    plot_panels(proj_tsne, "t-SNE", OUT_DIR / f"latent_space_tsne_{tag}.png")

    corr = np.corrcoef(proj[:, 0], mean_radial)[0, 1]
    print(f"corr(latent PC1, mean radial_distance) = {corr:.3f}")

    def between_var_frac(group_idx, n_groups):
        grand_mean = proj[:, 0].mean()
        between_var = sum(
            (proj[group_idx == g, 0].mean() - grand_mean) ** 2 * (group_idx == g).sum()
            for g in range(n_groups) if (group_idx == g).any()
        ) / len(proj)
        return between_var / proj[:, 0].var()

    # source-image ANOVA-style check: how much of PC1's variance is
    # "explained" by which image a subgraph came from, vs. by composition.
    img_frac = between_var_frac(source_idx, len(graphs))
    print(f"fraction of PC1 variance between source images (vs. within) = "
          f"{img_frac:.3f} (near 0 = latent space ignores image identity)")

    # Same check grouped by species (3 groups instead of len(graphs)) --
    # isolates species-level separation specifically, vs. per-image noise.
    species_frac = between_var_frac(dataset_idx, len(species_codes))
    print(f"fraction of PC1 variance between species (vs. within) = "
          f"{species_frac:.3f} across {len(species_codes)} species "
          f"({', '.join(f'{c}={DATASET_SPECIES.get(c, c)}' for c in species_codes)})")

    return {
        "corr_radial": corr,
        "between_image_var_frac": img_frac,
        "between_species_var_frac": species_frac,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npy"):
        evaluate(sys.argv[1])
    else:
        args = sys.argv[1:]
        variant = "norm" if "norm" in args else "raw"
        prefix = next((a for a in args if a != "norm"), None)
        graph_dir = out_dir_for(variant)
        pattern = f"{prefix}_*.pt" if prefix else "*.pt"
        tag = "_".join(t for t in (prefix, variant if variant == "norm" else None) if t) or "multi"
        graph_paths = sorted(graph_dir.glob(pattern))
        if not graph_paths:
            raise FileNotFoundError(
                f"no cached graphs matching {pattern!r} found in {graph_dir} -- run "
                f"`uv run python -m src.graph.build_all_graphs {variant}` first"
            )
        evaluate_multi(graph_paths, tag=tag)
