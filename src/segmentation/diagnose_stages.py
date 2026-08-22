"""Diagnostic visualization of every `segment_classical` pipeline stage on a
small, native-resolution patch -- individual cells should be visible by eye at
each stage, so problems (over/under-segmentation, broken wall detection,
spurious peaks) are attributable to a specific step rather than only visible
in the aggregate final result.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.classical_watershed import segment_classical
from src.segmentation.plantseg_eval import _load_sample

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"


def diagnose(sample_name: str = "DO_0000", box_frac=(0.40, 0.40, 0.48, 0.48), margin_px: int = 150, **kwargs):
    """`box_frac` = (x0, y0, x1, y1) as fractions of the full image, kept
    small (~8% of width/height) so individual cells are visible at native
    resolution in every panel -- the whole point of this diagnostic.

    Processes a `margin_px`-padded region and crops back down to `box_frac`
    afterwards, rather than segmenting the tight crop directly. Found via
    this diagnostic itself: segmenting a tight crop in isolation clips large
    lumens at the crop's own edge, so `tissue_mask`'s `binary_fill_holes`
    step can't fill them (they touch this crop's border, even though within
    the full image they're fully enclosed) -- confirmed by comparing a tight
    crop (96% tissue, visible holes) against the same region sliced out of a
    full-image-computed mask (100% tissue, no holes). Padding, not a
    tissue_mask code change, is the right fix -- the algorithm was correct,
    the crop-then-process order wasn't."""
    sample = _load_sample(sample_name)
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    x0, y0, x1, y1 = box_frac
    px0, py0, px1, py1 = int(w * x0), int(h * y0), int(w * x1), int(h * y1)

    pad_x0, pad_y0 = max(px0 - margin_px, 0), max(py0 - margin_px, 0)
    pad_x1, pad_y1 = min(px1 + margin_px, w), min(py1 + margin_px, h)
    padded_crop = img_full.crop((pad_x0, pad_y0, pad_x1, pad_y1))
    padded_rgb = np.array(padded_crop)
    print(f"{sample.name}: padded region {padded_rgb.shape[1]}x{padded_rgb.shape[0]} px (margin={margin_px}px)")

    labels_p, wall_mask_p, stages_p = segment_classical(padded_rgb, return_stages=True, **kwargs)

    # crop every stage array back down from the padded region to the
    # original box_frac window for display
    cy0, cx0 = py0 - pad_y0, px0 - pad_x0
    cy1, cx1 = cy0 + (py1 - py0), cx0 + (px1 - px0)

    def _crop(arr):
        return arr[cy0:cy1, cx0:cx1]

    rgb = _crop(padded_rgb)
    stages = {k: (_crop(v) if isinstance(v, np.ndarray) and v.ndim >= 2 else v) for k, v in stages_p.items()}
    # peak_coords are in padded-image coordinates; shift + filter to the display window
    coords_p = stages_p["peak_coords"]
    if len(coords_p):
        shifted = coords_p - np.array([cy0, cx0])
        in_view = (
            (shifted[:, 0] >= 0) & (shifted[:, 0] < (cy1 - cy0)) & (shifted[:, 1] >= 0) & (shifted[:, 1] < (cx1 - cx0))
        )
        stages["peak_coords"] = shifted[in_view]
    labels = _crop(labels_p)
    print(f"{sample.name} displayed patch: {rgb.shape[1]}x{rgb.shape[0]} px")

    n_instances = len(np.unique(labels)) - (1 if 0 in labels else 0)
    print(f"instances touching displayed patch: {n_instances}")

    panels = [
        ("original crop", rgb, None),
        ("tissue mask (fg_mask)", stages["fg_mask"], "gray"),
        ("inverted green channel", stages["inv"], "gray"),
        ("rolling-ball background", stages["background"], "gray"),
        ("background-corrected (u8)", stages["corrected_u8"], "gray"),
        ("wall mask (raw threshold)", stages["wall_mask_raw"], "gray"),
        ("wall mask (cleaned)", stages["wall_mask"], "gray"),
        ("interior (raw)", stages["interior_raw"], "gray"),
        ("interior (size-filtered)", stages["interior"], "gray"),
        ("distance transform", stages["distance"], "viridis"),
        ("distance (smoothed)", stages["distance_smooth"], "viridis"),
        ("watershed markers", None, None),  # special: distance_smooth + peak dots
        ("final instances", labels, "nipy_spectral"),
    ]

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.ravel()
    for ax, (title, arr, cmap) in zip(axes, panels):
        if title == "watershed markers":
            ax.imshow(stages["distance_smooth"], cmap="viridis")
            coords = stages["peak_coords"]
            if len(coords):
                ax.scatter(coords[:, 1], coords[:, 0], s=8, c="red", marker="x")
            ax.set_title(f"markers (n={len(coords)}) over smoothed distance")
        else:
            # explicit vmin/vmax for boolean masks -- imshow's auto-scaling
            # degenerates to solid black on a constant array (e.g. fg_mask
            # being uniformly True after the padding fix below), which reads
            # as "all excluded" when it actually means "all included".
            vmin, vmax = (0, 1) if isinstance(arr, np.ndarray) and arr.dtype == bool else (None, None)
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}{f' (n={n_instances})' if title == 'final instances' else ''}")
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{sample.name}_classical_stages_diagnostic.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")
    return labels, stages


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "DO_0000"
    diagnose(sample_name=name)
