"""Phase 2.5 — PlantSeg baseline via the underlying `pytorch3dunet` model code
(not the full PlantSeg app: no napari/vigra/Qt dependency). Loads Zenodo-hosted
pretrained 2D U-Net boundary-detection checkpoints (see
`models/pytorch3dunet/`, downloaded from the PlantSeg model zoo,
https://github.com/kreshuklab/plant-seg/blob/master/plantseg/resources/models_zoo.yaml)
and runs boundary prediction + a watershed instance step on the same test
crops used by `classical_watershed.py`, for a direct visual/quantitative
comparison.

Checkpoints (2D, boundary-prediction output, UNet2D architecture with
in_channels=1, out_channels=1, f_maps=64, layer_order='gcr', num_groups=8,
num_levels=4 -- confirmed by loading state_dicts with strict=True, not
assumed from doc-page config examples which turned out to reference a
different f_maps value):
  - confocal_2D_unet_ovules_ds2x: trained on confocal Arabidopsis ovule
    cross-sections. Closest available modality match to this project's
    stained brightfield shrub cross-sections (single 2D plane, not a
    Z-stack).
  - unet2d-lateral-root-lightsheet: trained on light-sheet root primordia
    images. Different modality; included per user request to compare both
    rather than assume which generalizes better.

**Polarity mismatch, found and corrected.** Both checkpoints were trained on
fluorescence images (bright cell walls on a dark background); this project's
stained brightfield images have the opposite polarity (dark walls on a light
background). Confirmed by direct test: `confocal_ovules` on this project's
data produced a near-zero boundary-probability map (max ~0.085, collapsing to
a single instance) on the raw grayscale crop, but a clear, wall-tracking
signal (max ~0.43) on the same crop with intensity inverted (`1 - gray`).
`lightsheet_root` happened to already produce a usable signal without
inversion -- not necessarily generalizing better, more likely incidentally
less polarity-sensitive; both checkpoints are configured with an explicit
per-model `invert` flag below rather than assuming one convention.

**Fixed threshold was also wrong.** A flat 0.5 boundary-probability cutoff
(reasonable for a well-calibrated in-domain model) silently collapsed
`confocal_ovules`'s ~0.085 max-probability output to nothing. Replaced with
a per-image Otsu threshold on the probability map, which adapts to each
model/image's actual output scale instead of assuming sigmoid outputs near
1.0 for confident boundaries out-of-domain.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, filters, morphology, segmentation
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pytorch3dunet.unet3d.model import UNet2D

from src.graph.features import extract_cell_features
from src.segmentation.classical_watershed import segment_classical, tissue_mask
from src.segmentation.do_dataset import load_test_split, load_train_split

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "pytorch3dunet"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase2"


def _load_sample(sample_name: str | None = None, sample_index: int = 0):
    """Resolves a DO sample either by name (searches both train/test splits
    -- e.g. `DO_0000` is in the train split, not test, so a name lookup
    can't just index one split) or by index into the test split (the
    original default before per-image selection was needed)."""
    if sample_name is None:
        return load_test_split()[sample_index]
    for sample in load_train_split() + load_test_split():
        if sample.name == sample_name:
            return sample
    raise ValueError(f"no DO sample named {sample_name!r} in train or test split")

CHECKPOINTS = {
    "confocal_ovules": MODELS_DIR / "confocal_2D_unet_ovules_ds2x.pytorch",
    "lightsheet_root": MODELS_DIR / "unet2d-lateral-root-lightsheet.pytorch",
}

# Both checkpoints trained on fluorescence data (bright walls on dark
# background); this project's stained brightfield crops are the opposite
# polarity. Determined empirically per-checkpoint (see module docstring),
# not assumed -- do not extend this dict without re-checking.
INVERT_INPUT = {
    "confocal_ovules": True,
    "lightsheet_root": False,
}


def load_model(checkpoint_path: Path, device: torch.device) -> UNet2D:
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def predict_boundaries(
    model: UNet2D,
    gray: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 192,
) -> np.ndarray:
    """Tiled inference over a grayscale image, matching the checkpoints'
    trained-on patch scale (recommended_patch_size [1, 256, 256] in the
    model zoo). Overlapping tiles (`stride` < `patch_size`) are averaged in
    the overlap region to avoid tile-boundary seams in the output."""
    h, w = gray.shape
    # global standardization, matching pytorch3dunet's `Standardize` transform
    mean, std = gray.mean(), gray.std()
    normed = (gray - mean) / max(std, 1e-6)

    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    padded = np.pad(normed, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    prob_sum = np.zeros((ph, pw), dtype=np.float64)
    weight = np.zeros((ph, pw), dtype=np.float64)

    ys = list(range(0, max(ph - patch_size, 0) + 1, stride))
    xs = list(range(0, max(pw - patch_size, 0) + 1, stride))
    if ys[-1] != ph - patch_size:
        ys.append(ph - patch_size)
    if xs[-1] != pw - patch_size:
        xs.append(pw - patch_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = padded[y : y + patch_size, x : x + patch_size]
                inp = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
                out = model(inp)
                pred = out.squeeze().cpu().numpy()
                prob_sum[y : y + patch_size, x : x + patch_size] += pred
                weight[y : y + patch_size, x : x + patch_size] += 1.0

    prob = prob_sum / np.maximum(weight, 1e-6)
    return prob[:h, :w]


def instances_from_boundary_prob(
    boundary_prob: np.ndarray,
    fg_mask: np.ndarray | None = None,
    min_cell_px: int = 15,
    peak_min_distance: int = 10,
    peak_h: float = 3.0,
):
    """Turn a boundary-probability map into instance labels via a
    distance-transform watershed -- same post-processing family as
    `classical_watershed.segment_classical`, so the two methods are
    compared on equal footing (the difference under test is the boundary
    *detector*, not the instance-splitting step).

    Threshold is Otsu on the probability map itself, not a fixed 0.5 --
    out-of-domain models produce low-confidence, uncalibrated probability
    scales (observed max ~0.085-0.43, not ~1.0), so a fixed cutoff can
    silently zero out an otherwise-usable signal.

    `fg_mask` restricts segmentation to actual tissue -- without it, the
    plain white slide background outside the section gets segmented into
    large meaningless "cells" (confirmed visually on a full-image run: the
    background split into a handful of giant flat polygons). Same fix
    `classical_watershed.tissue_mask` already needed.

    `peak_h` applies h-maxima suppression of shallow/spurious distance-map
    peaks before requiring surviving peaks to also be spatially separated
    (`peak_min_distance`) -- without it, a native-resolution full-image run
    showed real over-segmentation (a single visible cell lumen split into
    2-3 fragments on direct crop comparison against the source image, not
    just a downsampling artifact in a shrunk preview). Same fix
    `classical_watershed.segment_classical` already needed for the same
    reason (internal texture/noise creating multiple close local maxima per
    real cell)."""
    boundary_thresh = filters.threshold_otsu(boundary_prob)
    interior = boundary_prob < boundary_thresh
    if fg_mask is not None:
        interior &= fg_mask
    interior = ndi.binary_opening(interior, structure=np.ones((3, 3)))

    distance = ndi.distance_transform_edt(interior)
    distance_smooth = ndi.gaussian_filter(distance, sigma=2.0)

    hmax = h_maxima(distance_smooth, h=peak_h)
    coords = peak_local_max(distance_smooth, min_distance=peak_min_distance, labels=interior & (hmax > 0))
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    labels = segmentation.watershed(-distance_smooth, markers, mask=interior)

    # drop tiny fragments
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, labels.max() + 1))
    small = np.where(sizes < min_cell_px)[0] + 1
    labels[np.isin(labels, small)] = 0
    return labels


def boundary_mask_from_prob(boundary_prob: np.ndarray, thin: bool = True, line_width: int = 1) -> np.ndarray:
    """Otsu-thresholded binary boundary mask (see `instances_from_boundary_prob`
    docstring for why a fixed cutoff is wrong here). The raw thresholded mask
    is a blobby band several pixels wide; `thin=True` skeletonizes it down to
    a single-pixel-wide centerline so the overlay reads as boundary *lines*
    rather than filled regions. `line_width` optionally re-dilates the
    skeleton by a small, fixed amount (independent of the original probability
    map's blob thickness) for visibility at full-image scale."""
    thresh = filters.threshold_otsu(boundary_prob)
    mask = boundary_prob >= thresh
    if thin:
        mask = morphology.skeletonize(mask)
        if line_width > 1:
            mask = morphology.binary_dilation(mask, morphology.disk((line_width - 1) // 2))
    return mask


def overlay_boundaries(rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple, alpha: float = 0.65) -> np.ndarray:
    """Alpha-blend `color_rgb` onto `rgb` wherever `mask` is True."""
    out = rgb.astype(np.float64).copy()
    color_arr = np.array(color_rgb, dtype=np.float64)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return out.astype(np.uint8)


def full_image_overlay(
    sample_index: int = 0,
    sample_name: str | None = None,
    max_dim_full: int = 6000,
    preview_max_dim: int = 2400,
    line_width: int = 2,
):
    """Run both checkpoints over an entire (native-resolution) test image and
    save boundary overlays -- one per model, plus a combined overlay with
    both models' boundaries in distinct colors so agreement/disagreement is
    visible at a glance."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    sample = _load_sample(sample_name, sample_index)
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    print(f"{sample.name}: native size {w}x{h}")
    if max(w, h) > max_dim_full:
        scale = max_dim_full / max(w, h)
        img_full = img_full.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"downsized to {img_full.size} for tractable full-image inference")

    rgb = np.array(img_full)
    gray = color.rgb2gray(rgb).astype(np.float32)

    models = {name: load_model(path, device) for name, path in CHECKPOINTS.items()}
    overlay_colors = {
        "confocal_ovules": (255, 30, 30),  # red
        # not cyan: this project's images are safranin+astral-blue stained,
        # so the tissue's own cell walls are already blue -- a cyan overlay
        # camouflages against them. Yellow contrasts against both the blue
        # walls and the red overlay color above.
        "lightsheet_root": (255, 225, 0),  # yellow
    }

    masks = {}
    for name, model in models.items():
        print(f"running {name} over full image...")
        model_input = (1.0 - gray) if INVERT_INPUT[name] else gray
        boundary_prob = predict_boundaries(model, model_input, device)
        masks[name] = boundary_mask_from_prob(boundary_prob, thin=True, line_width=line_width)
        print(f"{name}: {masks[name].mean():.3%} of pixels marked as boundary (thinned)")

        single_overlay = overlay_boundaries(rgb, masks[name], overlay_colors[name], alpha=0.9)
        out_path = OUT_DIR / f"{sample.name}_full_overlay_{name}_thin.png"
        Image.fromarray(single_overlay).save(out_path)
        print(f"saved {out_path}")

    combined = rgb.copy()
    for name in models:
        combined = overlay_boundaries(combined, masks[name], overlay_colors[name], alpha=0.9)
    combined_path = OUT_DIR / f"{sample.name}_full_overlay_combined_thin.png"
    Image.fromarray(combined).save(combined_path)
    print(f"saved {combined_path}")

    if max(combined.shape[:2]) > preview_max_dim:
        preview_img = Image.fromarray(combined)
        scale = preview_max_dim / max(preview_img.size)
        preview_img = preview_img.resize(
            (int(preview_img.size[0] * scale), int(preview_img.size[1] * scale)), Image.LANCZOS
        )
        preview_path = OUT_DIR / f"{sample.name}_full_overlay_combined_thin_preview.png"
        preview_img.save(preview_path)
        print(f"saved {preview_path}")


def full_image_instance_maps(
    sample_index: int = 0,
    sample_name: str | None = None,
    max_dim_full: int = 6000,
    preview_max_dim: int = 2400,
):
    """Full-resolution per-cell instance segmentation (not just a boundary
    overlay) for each checkpoint -- runs the same
    `instances_from_boundary_prob` watershed step used in the crop
    comparison, but over the entire native-resolution test image, and
    renders each instance as a distinct random color (`label2rgb`), matching
    the `nipy_spectral`-style instance visualization used for
    `classical_watershed.py` elsewhere in this project."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    sample = _load_sample(sample_name, sample_index)
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    print(f"{sample.name}: native size {w}x{h}")
    if max(w, h) > max_dim_full:
        scale = max_dim_full / max(w, h)
        img_full = img_full.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"downsized to {img_full.size} for tractable full-image inference")

    rgb = np.array(img_full)
    gray = color.rgb2gray(rgb).astype(np.float32)

    print("computing tissue mask (restricts segmentation to actual tissue, not slide background)...")
    fg_mask = tissue_mask(rgb)

    models = {name: load_model(path, device) for name, path in CHECKPOINTS.items()}

    def save_full(arr: np.ndarray, path: Path, resample=Image.LANCZOS):
        Image.fromarray(arr).save(path)
        print(f"saved {path}")
        if max(arr.shape[:2]) > preview_max_dim:
            preview_img = Image.fromarray(arr)
            scale = preview_max_dim / max(preview_img.size)
            preview_img = preview_img.resize(
                (int(preview_img.size[0] * scale), int(preview_img.size[1] * scale)), resample
            )
            preview_path = path.with_name(path.stem + "_preview" + path.suffix)
            preview_img.save(preview_path)
            print(f"saved {preview_path}")

    for name, model in models.items():
        print(f"running {name} full-image instance segmentation...")
        model_input = (1.0 - gray) if INVERT_INPUT[name] else gray
        boundary_prob = predict_boundaries(model, model_input, device)
        labels = instances_from_boundary_prob(boundary_prob, fg_mask=fg_mask)
        n_instances = labels.max()
        print(f"{name}: {n_instances} instances (tissue-masked, h-maxima peak suppression)")

        colored = label2rgb(labels, image=None, bg_label=0, bg_color=(1, 1, 1))
        colored_u8 = (colored * 255).astype(np.uint8)
        save_full(colored_u8, OUT_DIR / f"{sample.name}_full_instances_{name}.png", resample=Image.NEAREST)

        # Mark each detected lumen (cell interior) with a distinct thin
        # outline on top of the original image -- separate from the
        # wall-boundary-line overlays produced by `full_image_overlay`, so
        # "this pixel is inside a detected cell" is visible on its own.
        lumen_outline = find_boundaries(labels, mode="outer")
        lumen_marked = overlay_boundaries(rgb, lumen_outline, (0, 255, 60), alpha=1.0)
        save_full(lumen_marked, OUT_DIR / f"{sample.name}_full_lumen_marked_{name}.png")

        # Every detected cell (lumen) filled solid black, walls/background
        # left as the original stained image -- shows all segmented cells
        # at a glance rather than just their outlines.
        lumen_fill = overlay_boundaries(rgb, labels > 0, (0, 0, 0), alpha=1.0)
        save_full(lumen_fill, OUT_DIR / f"{sample.name}_full_lumen_fill_{name}.png")


def full_image_cell_types(
    sample_index: int = 0,
    sample_name: str | None = None,
    model_name: str = "lightsheet_root",
    max_dim_full: int = 6000,
    preview_max_dim: int = 2400,
):
    """Color each detected cell by its growth-ring metadata class from
    `datasets/DO/annotations/*.tiff` (per-pixel integer ring index, -1 for
    unassigned/background), rather than an invented cell-type split.
    Confirmed by direct visualization that this annotation is the
    dendrochronological ring index (perfect concentric-ring pattern), not
    potato tissue-region labels as originally assumed early in this project
    -- see PLAN.md Phase 2.5 correction note. Uses `lightsheet_root` by
    default since it was the stronger boundary detector in this phase's
    earlier comparison."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    sample = _load_sample(sample_name, sample_index)
    img_full = Image.open(sample.image_path).convert("RGB")
    ann_full = np.array(Image.open(sample.annotation_path))
    w, h = img_full.size
    print(f"{sample.name}: native size {w}x{h}, annotation shape {ann_full.shape}")
    if max(w, h) > max_dim_full:
        scale = max_dim_full / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img_full = img_full.resize(new_size, Image.LANCZOS)
        # nearest-neighbor: annotation values are categorical ring indices,
        # not continuous intensities -- any smoothing resample would invent
        # nonexistent intermediate ring labels at boundaries.
        ann_full = np.array(Image.fromarray(ann_full).resize(new_size, Image.NEAREST))
        print(f"downsized to {new_size} for tractable full-image inference")

    rgb = np.array(img_full)
    gray = color.rgb2gray(rgb).astype(np.float32)
    fg_mask = tissue_mask(rgb)

    model = load_model(CHECKPOINTS[model_name], device)
    model_input = (1.0 - gray) if INVERT_INPUT[model_name] else gray
    print(f"running {model_name} full-image instance segmentation...")
    boundary_prob = predict_boundaries(model, model_input, device)
    labels = instances_from_boundary_prob(boundary_prob, fg_mask=fg_mask)
    n_instances = int(labels.max())
    print(f"{model_name}: {n_instances} instances")

    # Majority-vote the ring-index annotation under each cell instance,
    # vectorized (a per-instance Python loop over ~10k instances would be
    # slow): encode (label, ring_class) pairs as a single integer and
    # bincount, rather than looping.
    shifted_ann = (ann_full.astype(np.int64) + 1)  # -1 (unassigned) -> 0
    n_classes = int(shifted_ann.max()) + 1
    n_labels = n_instances + 1
    valid = labels > 0
    combined = labels[valid].astype(np.int64) * n_classes + shifted_ann[valid]
    counts = np.bincount(combined, minlength=n_labels * n_classes).reshape(n_labels, n_classes)
    majority_ring = counts.argmax(axis=1) - 1  # back to real ring index; -1 = unassigned/background region

    ring_values = sorted(set(majority_ring[1:].tolist()))
    print(f"ring classes present among detected cells: {ring_values}")

    cmap = plt.get_cmap("tab20")
    lut = np.zeros((n_labels, 3), dtype=np.float64)
    for lbl in range(1, n_labels):
        r = majority_ring[lbl]
        lut[lbl] = (0.55, 0.55, 0.55) if r < 0 else cmap(r % 20)[:3]

    color_per_pixel = (lut[labels] * 255).astype(np.uint8)
    out = rgb.copy()
    out[valid] = (0.15 * rgb[valid] + 0.85 * color_per_pixel[valid]).astype(np.uint8)
    boundaries = find_boundaries(labels, mode="outer")
    out[boundaries] = (0, 0, 0)

    out_path = OUT_DIR / f"{sample.name}_full_celltypes_{model_name}.png"
    Image.fromarray(out).save(out_path)
    print(f"saved {out_path}")
    if max(out.shape[:2]) > preview_max_dim:
        preview_img = Image.fromarray(out)
        scale = preview_max_dim / max(preview_img.size)
        preview_img = preview_img.resize(
            (int(preview_img.size[0] * scale), int(preview_img.size[1] * scale)), Image.LANCZOS
        )
        preview_path = out_path.with_name(out_path.stem + "_preview" + out_path.suffix)
        preview_img.save(preview_path)
        print(f"saved {preview_path}")


def _save_metric_colored(
    rgb: np.ndarray,
    labels: np.ndarray,
    metric: str,
    out_path: Path,
    cmap_name: str = "viridis",
    pct_clip: tuple = (2, 98),
    preview_max_dim: int = 2400,
) -> pd.DataFrame:
    """Shared rendering step for `full_image_cell_metrics` (U-Net checkpoints)
    and `full_image_classical_metrics` (`classical_watershed.segment_classical`)
    -- both produce an instance label array and then need the exact same
    "color each cell by a Phase 2 feature column" treatment, so this is
    factored out rather than duplicated per segmentation method (the point
    of coloring by the same metric is a fair, consistent comparison between
    methods)."""
    df = extract_cell_features(labels)
    if metric not in df.columns:
        raise ValueError(f"unknown metric {metric!r}; available: {sorted(df.columns)}")
    print(f"{metric}: {df[metric].describe()}")

    # note: extract_cell_features drops degenerate fragments (min_area/
    # min_minor_axis) -- those labels get no metric value and are left
    # uncolored (shown as plain original image) below, distinct from a
    # metric value of zero.
    lut = np.zeros(int(labels.max()) + 1, dtype=np.float64)
    lut[df["label"].to_numpy()] = df[metric].to_numpy()
    has_metric = np.isin(labels, df["label"].to_numpy())

    lo, hi = np.percentile(df[metric].to_numpy(), pct_clip)
    per_pixel_value = lut[labels]
    norm = np.clip((per_pixel_value - lo) / max(hi - lo, 1e-9), 0, 1)

    cmap = plt.get_cmap(cmap_name)
    color_per_pixel = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    out = rgb.copy()
    out[has_metric] = (0.15 * rgb[has_metric] + 0.85 * color_per_pixel[has_metric]).astype(np.uint8)
    boundaries = find_boundaries(labels, mode="outer")
    out[boundaries] = (0, 0, 0)

    Image.fromarray(out).save(out_path)
    print(f"saved {out_path} (color range: {metric} in [{lo:.2f}, {hi:.2f}], percentile clip {pct_clip})")
    if max(out.shape[:2]) > preview_max_dim:
        preview_img = Image.fromarray(out)
        scale = preview_max_dim / max(preview_img.size)
        preview_img = preview_img.resize(
            (int(preview_img.size[0] * scale), int(preview_img.size[1] * scale)), Image.LANCZOS
        )
        preview_path = out_path.with_name(out_path.stem + "_preview" + out_path.suffix)
        preview_img.save(preview_path)
        print(f"saved {preview_path}")
    return df


def full_image_cell_metrics(
    sample_index: int = 0,
    sample_name: str | None = None,
    model_name: str = "lightsheet_root",
    metric: str = "area",
    max_dim_full: int = 6000,
    preview_max_dim: int = 2400,
    cmap_name: str = "viridis",
    pct_clip: tuple = (2, 98),
):
    """Color each detected cell by a real per-cell geometry metric (area,
    elongation, circularity, eccentricity, radial_distance, ... -- see
    `src.graph.features.extract_cell_features`, Phase 2's existing feature
    table), not the ring-index ground-truth metadata used in
    `full_image_cell_types`. Reuses the Phase 2 feature extractor directly
    rather than recomputing region properties ad hoc, so this stays
    consistent with the feature definitions the rest of the pipeline (graph
    building, GVAE) already relies on.

    `pct_clip` percentile-clips the color scale -- a handful of huge vessel
    lumens would otherwise compress the color range for every ordinary
    fiber/parenchyma cell to a sliver near zero."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    sample = _load_sample(sample_name, sample_index)
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    print(f"{sample.name}: native size {w}x{h}")
    if max(w, h) > max_dim_full:
        scale = max_dim_full / max(w, h)
        img_full = img_full.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"downsized to {img_full.size} for tractable full-image inference")

    rgb = np.array(img_full)
    gray = color.rgb2gray(rgb).astype(np.float32)
    fg_mask = tissue_mask(rgb)

    model = load_model(CHECKPOINTS[model_name], device)
    model_input = (1.0 - gray) if INVERT_INPUT[model_name] else gray
    print(f"running {model_name} full-image instance segmentation...")
    boundary_prob = predict_boundaries(model, model_input, device)
    labels = instances_from_boundary_prob(boundary_prob, fg_mask=fg_mask)
    print(f"{model_name}: {int(labels.max())} instances")

    out_path = OUT_DIR / f"{sample.name}_full_metric_{metric}_{model_name}.png"
    _save_metric_colored(rgb, labels, metric, out_path, cmap_name, pct_clip, preview_max_dim)


def full_image_classical_metrics(
    sample_index: int = 0,
    sample_name: str | None = None,
    metric: str = "area",
    max_dim_full: int = 6000,
    preview_max_dim: int = 2400,
    cmap_name: str = "viridis",
    pct_clip: tuple = (2, 98),
):
    """Same area/elongation/etc. per-cell metric coloring as
    `full_image_cell_metrics`, but for `classical_watershed.segment_classical`
    instead of a U-Net checkpoint -- direct comparison point for "how does
    PlantSeg's boundary detector compare to the classical pipeline" on the
    exact same image and the exact same visualization/color scale."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = _load_sample(sample_name, sample_index)
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    print(f"{sample.name}: native size {w}x{h}")
    if max(w, h) > max_dim_full:
        scale = max_dim_full / max(w, h)
        img_full = img_full.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        print(f"downsized to {img_full.size} for tractable full-image processing")

    rgb = np.array(img_full)
    print("running classical watershed (segment_classical) on full image...")
    labels, _wall_mask = segment_classical(rgb)
    print(f"classical watershed: {int(labels.max())} instances")

    out_path = OUT_DIR / f"{sample.name}_full_metric_{metric}_classical.png"
    _save_metric_colored(rgb, labels, metric, out_path, cmap_name, pct_clip, preview_max_dim)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    sample = load_test_split()[0]
    img_full = Image.open(sample.image_path).convert("RGB")
    w, h = img_full.size
    # same crop framing as classical_watershed.py's periderm test, plus a
    # second interior/parenchyma-analogous crop for a second data point
    crops = {
        "periderm": img_full.crop((int(w * 0.05), int(h * 0.15), int(w * 0.20), int(h * 0.85))),
        "interior": img_full.crop((int(w * 0.40), int(h * 0.40), int(w * 0.60), int(h * 0.60))),
    }

    models = {name: load_model(path, device) for name, path in CHECKPOINTS.items()}

    for crop_name, crop_img in crops.items():
        rgb = np.array(crop_img)
        gray = color.rgb2gray(rgb).astype(np.float32)

        classical_labels, classical_wall_mask = segment_classical(rgb)

        n_cols = 2 + 2 * len(models)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{sample.name} {crop_name} crop")
        axes[1].imshow(classical_labels, cmap="nipy_spectral")
        axes[1].set_title(f"classical watershed\n(n={classical_labels.max()})")

        col = 2
        for name, model in models.items():
            print(f"[{crop_name}] running {name}...")
            model_input = (1.0 - gray) if INVERT_INPUT[name] else gray
            boundary_prob = predict_boundaries(model, model_input, device)
            unet_labels = instances_from_boundary_prob(boundary_prob)
            n_instances = unet_labels.max()
            print(f"[{crop_name}] {name}: {n_instances} instances")

            axes[col].imshow(boundary_prob, cmap="magma")
            axes[col].set_title(f"{name}\nboundary prob.")
            axes[col + 1].imshow(unet_labels, cmap="nipy_spectral")
            axes[col + 1].set_title(f"{name}\ninstances (n={n_instances})")
            col += 2

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_path = OUT_DIR / f"{sample.name}_{crop_name}_plantseg_vs_classical.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    name_arg = None
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        name_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    if "--classical-metrics" in sys.argv:
        idx = sys.argv.index("--classical-metrics")
        metric_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "area"
        full_image_classical_metrics(metric=metric_arg, sample_name=name_arg)
    elif "--metrics" in sys.argv:
        idx = sys.argv.index("--metrics")
        metric_arg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "area"
        full_image_cell_metrics(metric=metric_arg, sample_name=name_arg)
    elif "--celltypes" in sys.argv:
        full_image_cell_types(sample_name=name_arg)
    elif "--instances" in sys.argv:
        full_image_instance_maps(sample_name=name_arg)
    elif "--full" in sys.argv:
        full_image_overlay(sample_name=name_arg)
    else:
        main()
