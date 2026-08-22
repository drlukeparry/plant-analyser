"""QC overlays: color cells by feature to sanity-check against visible tissue rings."""
import numpy as np


def paint_by_feature(labels: np.ndarray, node_df, feature: str) -> np.ndarray:
    """Return a float image where each cell's pixels are set to its feature value."""
    lut = np.zeros(labels.max() + 1, dtype=np.float64)
    lut[node_df["label"].values] = node_df[feature].values
    return lut[labels]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.graph.build_graph import build_graph

    labels = np.load(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/phase2/qc_overlay.png")

    data, node_df = build_graph(labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, feature, cmap in zip(
        axes, ["elongation", "orientation", "radial_distance"], ["viridis", "twilight", "plasma"]
    ):
        img = paint_by_feature(labels, node_df, feature)
        im = ax.imshow(np.ma.masked_where(labels == 0, img), cmap=cmap)
        ax.set_title(feature)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
