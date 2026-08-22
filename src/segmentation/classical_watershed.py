"""Classical rolling-ball + adaptive-threshold + watershed segmentation.

Test of the potato-tuber dataset paper's own ground-truth generation approach
(background correction, thresholding, morphological cleanup) as a fix for the
periderm band where Cellpose-SAM's zero-shot flow-based segmentation fails
(see PLAN.md Phase 1 known issue).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology, restoration, segmentation
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import h_maxima
from skimage.util import img_as_ubyte

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.do_dataset import load_train_split

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"


def tissue_mask(
    rgb: np.ndarray,
    min_region_px: int = 5000,
    sat_thresh: float = 0.08,
    val_thresh: float = 0.85,
) -> np.ndarray:
    """Foreground (tissue) vs. background (white slide) mask, cleaned up with
    morphology. Without this, watershed run on the flat white background
    produces meaningless giant/degenerate "cells" with extreme feature values
    (seen in Phase 2 QC — elongation up to ~1e8 on background blobs).

    Uses `binary_fill_holes`, not `remove_small_holes` -- large vessel-element
    lumens are optically as bright as the slide background (both read as
    "not tissue" under a plain darkness threshold), but `remove_small_holes`
    only fills holes below `area_threshold`, so big vessels stayed excluded
    from the mask (confirmed visually: full-image instance runs left every
    large vessel lumen unlabeled while correctly labeling smaller
    fiber/parenchyma lumens). `binary_fill_holes` fills *any* hole not
    connected to the image border -- which correctly includes vessels of any
    size (they're always fully enclosed by tissue) while still leaving the
    real background alone (it's the one region actually touching the
    border).

    Classifies on HSV saturation + value, not a grayscale Otsu split.
    Diagnosed via `diagnose_stages.py` on a small tissue-only patch: Otsu
    forces a bimodal split of whatever brightness values are present, so on
    a crop containing no real background at all it still split the crop's
    own brightness range in half, marking real tissue corners as
    "background" (confirmed: real background samples measured
    saturation~0.025/value~0.96, vs. this crop's tissue at
    saturation~0.166/value up to 1.0 -- Otsu-on-grayscale ignores the color
    channel entirely, so it can't see that distinction, and the "threshold"
    it picks depends on whatever's in the given image/crop rather than an
    absolute, transferable property of real background vs. tissue).
    Saturation is scale-invariant: real slide background is reliably
    colorless regardless of crop size, while even the brightest cell lumen
    carries a faint stain tint. `val_thresh` additionally requires brightness
    (background is bright *and* colorless -- a stray dark low-saturation
    pixel, e.g. deep shadow, shouldn't count as background either)."""
    hsv = color.rgb2hsv(rgb)
    saturation, value = hsv[..., 1], hsv[..., 2]
    mask = ~((saturation < sat_thresh) & (value > val_thresh))  # tissue = not (colorless and bright)
    mask = morphology.remove_small_objects(mask, min_size=min_region_px)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    return mask


def segment_classical(
    rgb: np.ndarray,
    rolling_ball_radius: int = 25,
    min_cell_px: int = 15,
    peak_h_frac: float = 0.7,
    peak_min_distance: int = 8,
    distance_smooth_sigma: float = 2.0,
    return_stages: bool = False,
):
    """`peak_h_frac`/`peak_min_distance`/`distance_smooth_sigma` control
    watershed over-segmentation: internal wall-color gradients and
    cell-interior texture otherwise create multiple close local maxima per
    real cell, splitting one cell into many fragments (observed: ~504
    watershed regions in a crop with ~15-20 real cells). Smoothing the
    distance map and requiring peaks to be both well-separated
    (`peak_min_distance`) and prominent merges those into single per-cell
    markers.

    `peak_h_frac` is a *relative* (per-cell) h-maxima threshold, not an
    absolute pixel height -- found via `diagnose_stages.py` that a single
    absolute `peak_h` (the original approach) cannot serve both small and
    large cells at once: high enough to suppress noise in small cells, and
    it silently drops ~2/3 of real small-cell markers entirely (confirmed:
    at the old peak_h=4.0, only 62/192 real-cell-sized interior regions in a
    test patch got a marker at all -- not merged, just missing); low enough
    to recover those, and internal texture/gradient bumps inside large cells
    now clear the same absolute bar too, fragmenting one real large cell
    into several (confirmed: a single 18,650px vessel-sized cell split into
    5 pieces at a recovery-strength absolute-equivalent threshold). Fixed by
    normalizing each connected interior component's distance-transform
    values by *that component's own max* before applying `h_maxima` with a
    single relative threshold: a small cell's own peak is always ~1.0 in
    its own normalized frame (so it survives regardless of absolute pixel
    size), while a large cell's internal texture bumps must clear the same
    *fraction* of that specific cell's own radius to count as a separate
    peak -- texture noise usually doesn't, while two genuinely
    comparably-sized sub-peaks still would. Verified empirically at
    `peak_h_frac=0.7`: recovers small-cell markers (64->307 final instances
    on a small-cell test patch, matching the aggressive-but-over-splitting
    fix) while keeping the 18,650px test vessel as a single label (vs. 5 at
    a too-permissive relative threshold). This is a targeted fix, not the
    same as the earlier-attempted (and reverted) spatial local-scale map --
    that used a smoothed neighbor-based size *estimate*, which misjudges
    atypically-sized cells; this uses each component's own exact max, no
    estimation involved.

    `return_stages=True` additionally returns every intermediate array as a
    dict, for diagnosing the pipeline stage-by-stage (see
    `diagnose_stages.py`) rather than only inspecting the final result.
    """
    fg_mask = tissue_mask(rgb)

    # Wall stain is strong in the blue/red channels for this stain; use inverted
    # green channel as a proxy for "wall intensity" (cell walls are darker in green).
    gray = rgb[:, :, 1].astype(np.float64)
    inv = 255 - gray

    background = restoration.rolling_ball(inv, radius=rolling_ball_radius)
    corrected = inv - background
    corrected = np.clip(corrected, 0, None)
    corrected_u8 = img_as_ubyte(corrected / max(corrected.max(), 1e-6))

    # Local Otsu (vs. the earlier fixed-offset local-mean threshold) — self
    # calibrating per neighborhood with no hand-tuned offset constant, and
    # visually more consistent across both periderm and parenchyma regions
    # than global Otsu (illumination-sensitive) or Li's method (broken walls).
    local_thresh = rank.otsu(corrected_u8, morphology.disk(15))
    wall_mask_raw = corrected_u8 > local_thresh
    wall_mask = morphology.remove_small_objects(wall_mask_raw, min_size=8)
    wall_mask = morphology.binary_closing(wall_mask, morphology.disk(1))

    # Cell interiors are the complement of the wall mask, restricted to tissue.
    interior_raw = ~wall_mask & fg_mask
    interior = morphology.remove_small_objects(interior_raw, min_size=min_cell_px)

    distance = ndi.distance_transform_edt(interior)
    distance_smooth = ndi.gaussian_filter(distance, sigma=distance_smooth_sigma)

    # Per-component relative h-maxima (see docstring): normalize each
    # connected interior blob's distance values by that blob's own peak
    # before suppressing shallow maxima, so the suppression threshold scales
    # with each cell's own size instead of being a fixed pixel height.
    labeled_interior, n_interior_comp = ndi.label(interior)
    comp_max = ndi.maximum(distance_smooth, labeled_interior, index=np.arange(1, n_interior_comp + 1))
    comp_max_map = np.zeros_like(distance_smooth)
    nonzero = labeled_interior > 0
    comp_max_map[nonzero] = comp_max[labeled_interior[nonzero] - 1]
    normalized_distance = np.zeros_like(distance_smooth)
    normalized_distance[nonzero] = distance_smooth[nonzero] / np.maximum(comp_max_map[nonzero], 1e-6)

    hmax = h_maxima(normalized_distance, h=peak_h_frac)
    coords = peak_local_max(distance_smooth, min_distance=peak_min_distance, labels=interior & (hmax > 0))
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    labels = segmentation.watershed(-distance_smooth, markers, mask=interior)

    if not return_stages:
        return labels, wall_mask

    stages = {
        "fg_mask": fg_mask,
        "inv": inv,
        "background": background,
        "corrected_u8": corrected_u8,
        "wall_mask_raw": wall_mask_raw,
        "wall_mask": wall_mask,
        "interior_raw": interior_raw,
        "interior": interior,
        "distance": distance,
        "distance_smooth": distance_smooth,
        "hmax": hmax,
        "peak_coords": coords,
        "markers": markers,
        "labels": labels,
    }
    return labels, wall_mask, stages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_train_split()[0]
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    # same periderm crop used in the earlier native-res Cellpose test
    crop = img_full.crop((int(w * 0.05), int(h * 0.15), int(w * 0.20), int(h * 0.85)))
    rgb = np.array(crop)

    labels, wall_mask = segment_classical(rgb)
    n_instances = labels.max()
    print(f"classical watershed instances in periderm crop: {n_instances}")
    print(f"nonzero (assigned) fraction: {(labels > 0).mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    axes[0].imshow(rgb)
    axes[0].set_title("native-res periderm crop")
    axes[1].imshow(wall_mask, cmap="gray")
    axes[1].set_title("detected wall mask")
    axes[2].imshow(labels, cmap="nipy_spectral")
    axes[2].set_title(f"watershed instances (n={n_instances})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path = OUT_DIR / "DO_0000_periderm_classical_watershed.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
