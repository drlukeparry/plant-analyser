"""Phase 3 diagnostic: with species identity ruled out as the driver of the
GVAE latent space's between-image variance (2.7-6.7% between species vs.
35-44% between individual images, see PLAN.md), does per-image imaging or
segmentation variation explain it instead -- stain intensity/color, image
resolution, or segment_classical's cell-density/size sensitivity across very
different specimens?

For each of the 213 cached graphs, computes a per-image mean latent vector
(averaged over several sampled subgraphs, not just one) and a set of cheap
per-image imaging/segmentation covariates, then correlates each covariate
against the per-image latent position.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.gvae.eval import pca_2d
from src.models.gvae.model import GraphVAE
from src.models.gvae.sampling import sample_subgraph
from src.graph.build_all_graphs import out_dir_for
from src.models.gvae.train import device

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"

CANDIDATES = [
    "megapixels", "mean_r", "mean_g", "mean_b", "mean_saturation", "mean_value",
    "num_cells", "mean_area_px", "mean_equiv_diam_px", "cell_density",
]


def image_stats(name: str) -> dict:
    """Cheap per-image imaging statistics from a downsized thumbnail (color/
    resolution don't need full-res pixels to estimate)."""
    code = name.split("_")[0]
    path = DATASETS_ROOT / code / "inputimages" / f"{name}.jpg"
    im = Image.open(path)
    w, h = im.size
    im.thumbnail((512, 512))
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255.0
    return {
        "width": w, "height": h, "megapixels": w * h / 1e6,
        "mean_r": rgb[..., 0].mean(), "mean_g": rgb[..., 1].mean(), "mean_b": rgb[..., 2].mean(),
        "mean_saturation": hsv[..., 1].mean(), "mean_value": hsv[..., 2].mean(),
    }


def main(tag: str = "multi", variant: str = "raw", n_per_image: int = 20, num_hops: int = 3,
         max_nodes: int = 300):
    dev = device()
    graph_paths = sorted(out_dir_for(variant).glob("*.pt"))
    names = [p.stem for p in graph_paths]
    print(f"loading {len(graph_paths)} cached graphs ...")
    graphs, node_dfs = [], []
    for p in graph_paths:
        cached = torch.load(p, weights_only=False)
        graphs.append(cached["data"])
        node_dfs.append(cached["node_df"])

    scaler = np.load(OUT_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    graphs_std = [g.clone() for g in graphs]
    for g, gs in zip(graphs, graphs_std):
        gs.x = (g.x - mean) / std

    model = GraphVAE(in_dim=graphs[0].x.size(1)).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / f"gvae_{tag}.pt", map_location=dev))
    model.eval()

    print(f"sampling {n_per_image} subgraphs/image for a per-image mean latent ...")
    per_image_mean_z = []
    with torch.no_grad():
        for gi, g in enumerate(graphs_std):
            zs = []
            for _ in range(n_per_image):
                sub = sample_subgraph(g, num_hops=num_hops, max_nodes=max_nodes)
                if sub.num_nodes < 8:
                    continue
                batch = torch.zeros(sub.num_nodes, dtype=torch.long)
                mu, _ = model.encode(sub.x.to(dev), sub.edge_index.to(dev), batch.to(dev))
                zs.append(mu.cpu().numpy()[0])
            per_image_mean_z.append(np.mean(zs, axis=0) if zs else np.zeros(model.latent_dim))
            if (gi + 1) % 50 == 0:
                print(f"  {gi+1}/{len(graphs_std)} images done")
    per_image_mean_z = np.array(per_image_mean_z)

    # PCA computed directly on per-image means (213 points), not reused from
    # the per-subgraph run -- this is the natural axis for an image-level analysis.
    proj = pca_2d(per_image_mean_z)

    print("computing per-image imaging + segmentation stats ...")
    rows = []
    for name, node_df in zip(names, node_dfs):
        stats = image_stats(name)
        stats["name"] = name
        stats["species"] = name.split("_")[0]
        stats["num_cells"] = len(node_df)
        stats["mean_area_px"] = node_df["area"].mean()
        stats["mean_equiv_diam_px"] = node_df["equivalent_diameter"].mean()
        stats["cell_density"] = len(node_df) / (stats["width"] * stats["height"])
        rows.append(stats)
    df = pd.DataFrame(rows)
    df["latent_pc1"] = proj[:, 0]
    df["latent_pc2"] = proj[:, 1]

    out_csv = OUT_DIR / f"image_variance_diagnostic_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    print("\ncorrelation of per-image latent PC1 with imaging/segmentation covariates:")
    corrs = {}
    for c in CANDIDATES:
        r = np.corrcoef(df[c], df["latent_pc1"])[0, 1]
        corrs[c] = r
        print(f"  {c:20s} corr = {r:+.3f}")

    X = df[CANDIDATES].values
    X = (X - X.mean(0)) / X.std(0)
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, df["latent_pc1"].values, rcond=None)
    pred = X @ coef
    y = df["latent_pc1"].values
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\njoint R^2 of all imaging/segmentation covariates predicting per-image latent PC1: {r2:.3f}")

    import matplotlib.pyplot as plt

    top_covariate = max(corrs, key=lambda c: abs(corrs[c]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    palette = plt.get_cmap("tab10").colors
    for i, code in enumerate(sorted(df["species"].unique())):
        m = df["species"] == code
        axes[0].scatter(df.loc[m, top_covariate], df.loc[m, "latent_pc1"], s=14, alpha=0.8,
                         color=palette[i % 10], label=code)
    axes[0].set_xlabel(top_covariate)
    axes[0].set_ylabel("per-image mean latent PC1")
    axes[0].set_title(f"strongest covariate: {top_covariate} (r={corrs[top_covariate]:+.3f})")
    axes[0].legend(fontsize=8)

    order = sorted(corrs, key=lambda c: abs(corrs[c]), reverse=True)
    axes[1].barh(order, [corrs[c] for c in order])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("corr(covariate, per-image latent PC1)")
    fig.tight_layout()
    out_path = OUT_DIR / f"image_variance_diagnostic_{tag}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    return df


if __name__ == "__main__":
    # tag identifies which checkpoint/scaler to load (e.g. "multi", "norm",
    # "DO"); variant selects which graph cache to sample subgraphs from
    # (must match how that checkpoint was trained).
    tag = sys.argv[1] if len(sys.argv) > 1 else "multi"
    variant = sys.argv[2] if len(sys.argv) > 2 else ("norm" if tag == "norm" else "raw")
    main(tag=tag, variant=variant)
