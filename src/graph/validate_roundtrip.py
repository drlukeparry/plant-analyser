"""Phase 4 exit criterion: real graph -> field -> reconstructed graph, compare
node feature distributions and graph statistics. Don't proceed to Phase 5
(DDIM) until this is close."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_graph import build_graph
from src.graph.features import extract_cell_features
from src.graph.field import rasterize_field
from src.graph.tessellate import tessellate_from_field

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase4"
COMPARE_FEATURES = ["area", "elongation", "orientation", "radial_distance"]


def pct_shift(a: np.ndarray, b: np.ndarray) -> float:
    """% difference in means, relative to original's mean magnitude."""
    return 100 * abs(np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-9)


def validate(labels_path: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.load(labels_path)
    node_df = extract_cell_features(labels)
    field = rasterize_field(labels, node_df)
    recovered = tessellate_from_field(field)
    recovered_df = extract_cell_features(recovered)

    print(f"original: {len(node_df)} cells, recovered: {len(recovered_df)} cells "
          f"({100*(len(recovered_df)-len(node_df))/len(node_df):+.1f}%)")

    orig_data, _ = build_graph(labels)
    rec_data, _ = build_graph(recovered)
    orig_deg = orig_data.num_edges / orig_data.num_nodes
    rec_deg = rec_data.num_edges / rec_data.num_nodes
    print(f"avg degree: original={orig_deg:.2f}, recovered={rec_deg:.2f} "
          f"({100*(rec_deg-orig_deg)/orig_deg:+.1f}%)")

    fig, axes = plt.subplots(1, len(COMPARE_FEATURES), figsize=(5 * len(COMPARE_FEATURES), 4))
    report = {}
    for ax, feat in zip(axes, COMPARE_FEATURES):
        a, b = node_df[feat].values, recovered_df[feat].values
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, alpha=0.5, label="original", density=True)
        ax.hist(b, bins=bins, alpha=0.5, label="recovered", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
        shift = pct_shift(a, b)
        report[feat] = shift
        print(f"  {feat}: mean shift = {shift:.1f}%")
    fig.tight_layout()
    out_path = OUT_DIR / "roundtrip_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

    max_shift = max(report.values())
    print(f"\nmax feature mean shift: {max_shift:.1f}% "
          f"({'PASS' if max_shift < 10 else 'FAIL'} vs. 10% threshold from PLAN.md)")
    return report


if __name__ == "__main__":
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase1/DO_0000_full_labels.npy"
    validate(labels_path)
