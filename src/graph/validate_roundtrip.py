"""Phase 4 exit criterion: real graph -> field -> reconstructed graph, compare
node feature distributions and graph statistics. Don't proceed to Phase 5
(DDIM) until this is close.

Runs three conditions, not just one:
- exact: the original test -- labels -> noise-free field -> tessellate.
  Recovers markers from an exact, noise-free SDF (the field encodes the
  *exact* original boundary), which is a much easier problem than
  recovering markers from something a trained model actually generated --
  a near-0% shift here mostly proves the watershed inverse is deterministic,
  not that the field representation survives being generated.
- degraded: same field, blurred/quantized/noised to approximate what a
  trained UNet/DDIM's output would actually look like in Phase 5 -- this is
  the condition that matters for deciding whether the field representation
  is Phase-5-ready.
- patch: the degraded round trip repeated on 256x256 crops (Phase 5's
  actual training unit) rather than the whole image, since cells cut at
  patch boundaries are a real edge case a whole-image test never exercises.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_graph import build_graph
from src.graph.features import extract_cell_features
from src.graph.field import rasterize_field
from src.graph.tessellate import tessellate_from_field

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase4"
COMPARE_FEATURES = ["area", "elongation", "orientation", "radial_distance"]
# In std units (effect_size_shift), not % of mean -- see effect_size_shift's
# docstring for why a %-of-mean threshold is unusable for near-zero-mean
# features like orientation. 0.5 std is a moderate-effect-size cutoff
# (Cohen's convention), not independently derived from this project's data.
SHIFT_THRESHOLD = 0.5


def effect_size_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean difference relative to the *original's std*, not its mean --
    `pct_shift` blows up (divides by a near-zero denominator) for features
    like `orientation`, whose true mean is near zero by symmetry (cells
    point radially outward roughly as often as inward), so a tiny, harmless
    absolute difference reads as a huge, alarming percentage. This is the
    metric actually used for the pass/fail verdict; pct_shift is still
    printed alongside it for features where it's meaningful (area,
    radial_distance -- means well away from zero)."""
    return abs(np.mean(b) - np.mean(a)) / (np.std(a) + 1e-9)


def pct_shift(a: np.ndarray, b: np.ndarray) -> float:
    """% difference in means, relative to original's mean magnitude."""
    return 100 * abs(np.mean(b) - np.mean(a)) / (abs(np.mean(a)) + 1e-9)


def degrade_field(field: np.ndarray, blur_sigma: float = 1.5, noise_frac: float = 0.05,
                   quant_levels: int = 32, seed: int = 0) -> np.ndarray:
    """Approximates a trained UNet/DDIM's generated field: blurred (limited
    spatial precision), quantized (limited output precision), and noisy --
    vs. the exact field a real label map's SDF gives. Degradation strength
    is illustrative, not calibrated against an actual trained model (none
    exists yet at this phase) -- the point is to stop testing the noise-free
    case exclusively, not to claim these are the *right* numbers.
    """
    rng = np.random.default_rng(seed)
    degraded = np.empty_like(field)
    for c in range(field.shape[-1]):
        ch = ndi.gaussian_filter(field[..., c], sigma=blur_sigma)
        span = ch.max() - ch.min()
        if span > 1e-6:
            levels = np.round((ch - ch.min()) / span * (quant_levels - 1))
            ch = levels / (quant_levels - 1) * span + ch.min()
            ch = ch + rng.normal(0, noise_frac * span, ch.shape)
        degraded[..., c] = ch
    return degraded.astype(np.float32)


def _round_trip_stats(labels: np.ndarray, node_df, field: np.ndarray, name: str, out_dir: Path):
    recovered = tessellate_from_field(field)
    recovered_df = extract_cell_features(recovered)
    if len(recovered_df) == 0 or len(node_df) == 0:
        print(f"  [{name}] recovered 0 cells -- skipping (degenerate patch/field)")
        return None

    cell_count_shift = 100 * (len(recovered_df) - len(node_df)) / len(node_df)
    print(f"  [{name}] original: {len(node_df)} cells, recovered: {len(recovered_df)} cells "
          f"({cell_count_shift:+.1f}%)")

    orig_data, _ = build_graph(labels)
    rec_data, _ = build_graph(recovered)
    orig_deg = orig_data.num_edges / max(orig_data.num_nodes, 1)
    rec_deg = rec_data.num_edges / max(rec_data.num_nodes, 1)
    deg_shift = 100 * (rec_deg - orig_deg) / max(orig_deg, 1e-9)
    print(f"  [{name}] avg degree: original={orig_deg:.2f}, recovered={rec_deg:.2f} ({deg_shift:+.1f}%)")

    report = {}
    for feat in COMPARE_FEATURES:
        a, b = node_df[feat].values, recovered_df[feat].values
        shift = pct_shift(a, b)
        effect = effect_size_shift(a, b)
        report[feat] = effect
        print(f"  [{name}] {feat}: mean shift = {shift:.1f}% of mean, "
              f"{effect:.2f} std ({'PASS' if effect < SHIFT_THRESHOLD else 'FAIL'} "
              f"vs. {SHIFT_THRESHOLD:.1f} std threshold)")

    fig, axes = plt.subplots(1, len(COMPARE_FEATURES), figsize=(5 * len(COMPARE_FEATURES), 4))
    for ax, feat in zip(axes, COMPARE_FEATURES):
        a, b = node_df[feat].values, recovered_df[feat].values
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins=bins, alpha=0.5, label="original", density=True)
        ax.hist(b, bins=bins, alpha=0.5, label="recovered", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle(f"round-trip validation -- {name}")
    fig.tight_layout()
    out_path = out_dir / f"roundtrip_validation_{name}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  [{name}] saved {out_path}")

    max_shift = max(report.values())
    print(f"  [{name}] max feature effect-size shift: {max_shift:.2f} std "
          f"({'PASS' if max_shift < SHIFT_THRESHOLD else 'FAIL'} vs. {SHIFT_THRESHOLD:.1f} std threshold)\n")
    report["max_shift"] = max_shift
    report["cell_count_shift"] = cell_count_shift
    report["avg_degree_shift"] = deg_shift
    return report


def validate(labels_path: str, patch_size: int = 256, n_patches: int = 6, seed: int = 0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.load(labels_path)
    node_df = extract_cell_features(labels)
    field = rasterize_field(labels, node_df)

    results = {}

    print("=== exact (noise-free field, whole image) ===")
    results["exact"] = _round_trip_stats(labels, node_df, field, "exact", OUT_DIR)

    print("=== degraded (blurred/quantized/noised field, whole image -- approximates Phase 5 output) ===")
    results["degraded"] = _round_trip_stats(labels, node_df, degrade_field(field, seed=seed), "degraded", OUT_DIR)

    print(f"=== patch (degraded field, {n_patches}x {patch_size}px crops -- Phase 5's actual training unit) ===")
    rng = np.random.default_rng(seed)
    h, w = labels.shape
    patch_reports = []
    for i in range(n_patches):
        if h <= patch_size or w <= patch_size:
            print(f"  image smaller than patch_size={patch_size}, skipping patch test")
            break
        r0 = rng.integers(0, h - patch_size)
        c0 = rng.integers(0, w - patch_size)
        patch_labels = labels[r0:r0 + patch_size, c0:c0 + patch_size]
        patch_node_df = extract_cell_features(patch_labels)
        if len(patch_node_df) < 3:
            continue
        patch_field = rasterize_field(patch_labels, patch_node_df)
        r = _round_trip_stats(
            patch_labels, patch_node_df, degrade_field(patch_field, seed=seed + i),
            f"patch{i}", OUT_DIR,
        )
        if r is not None:
            patch_reports.append(r)
    if patch_reports:
        avg_max_shift = np.mean([r["max_shift"] for r in patch_reports])
        n_patch_pass = sum(r["max_shift"] < SHIFT_THRESHOLD for r in patch_reports)
        print(f"  patch test: {len(patch_reports)}/{n_patches} usable patches, "
              f"{n_patch_pass}/{len(patch_reports)} individually pass, "
              f"mean max-feature-shift = {avg_max_shift:.2f} std "
              f"({'PASS' if avg_max_shift < SHIFT_THRESHOLD else 'FAIL'} vs. {SHIFT_THRESHOLD:.1f} std threshold)")
    results["patches"] = patch_reports

    return results


if __name__ == "__main__":
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase1/DO_0000_full_labels.npy"
    validate(labels_path)
