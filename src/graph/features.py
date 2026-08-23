"""Per-cell geometry features from an instance label image (Phase 2)."""
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

PROPS = [
    "label",
    "area",
    "perimeter",
    "equivalent_diameter_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
    "eccentricity",
    "centroid",
]

_UM_COLS = [
    "area_um2", "perimeter_um", "equivalent_diameter_um",
    "major_axis_length_um", "minor_axis_length_um", "radial_distance_um",
]
_NORM_COLS = [
    "area_norm", "perimeter_norm", "equivalent_diameter_norm",
    "major_axis_length_norm", "minor_axis_length_norm", "radial_distance_norm",
]


def extract_cell_features(
    labels: np.ndarray,
    min_area: int = 10,
    min_minor_axis: float = 1.5,
    max_area_log_std: float | None = 3.0,
    um_per_pixel: float | None = None,
    normalize_by_image_size: bool = False,
) -> pd.DataFrame:
    """Per-cell shape/size/orientation features, plus radial position relative
    to the whole section's centroid (natural coordinate system for a stem/tuber
    cross-section, where tissue identity is strongly a function of radial position).

    Drops degenerate fragments (near-zero minor axis, e.g. 1px-wide slivers along
    tissue tears/cracks) that otherwise blow up elongation to nonsensical values —
    these are segmentation noise regardless of source, not real cells.

    `max_area_log_std` drops the opposite failure mode: found via
    `diagnose_stages.py` on `DO_0024` that genuinely damaged/torn tissue
    regions (faded, out-of-focus specimen tears -- confirmed visually, not
    an artifact of any particular segmentation method) have no real wall
    signal for the watershed to split on, so the whole degraded patch merges
    into one giant "cell" (observed: up to 138,518px vs. a ~1,375px 98th
    percentile for real cells in the same image -- ~100x larger). Filtered by
    excluding any cell whose area exceeds `exp(mean(log(area)) +
    max_area_log_std * std(log(area)))`. Uses *log*-area, not raw area, for
    the mean/std -- cell area is a strictly-positive, heavy-tailed quantity
    (real sizes span small fibers to large vessels multiplicatively, not
    additively), so raw-space std is dominated by, and unstable because of,
    exactly the extreme outliers this is trying to exclude; log-space makes
    "3 standard deviations" a stable, meaningful cutoff instead of one that
    moves around depending on how extreme the worst outlier happens to be.
    Set to `None` to disable (e.g. if a dataset is known to be damage-free).

    If `um_per_pixel` is given, adds physical-unit columns (`*_um`, `*_um2`)
    alongside the pixel-unit ones — see DO_UM_PER_PIXEL in
    src/segmentation/do_dataset.py for the derivation/caveats for datasets/DO.
    Ratios (elongation, circularity, solidity, eccentricity) and angles are
    unaffected by pixel size and aren't duplicated.

    If `normalize_by_image_size` is True, adds `*_norm` columns: size-related
    columns divided by a characteristic *tissue* scale (the 99th-percentile
    `radial_distance` among that image's own cells, i.e. roughly the
    specimen's own radius) instead of an absolute pixel count. This is the
    practical substitute for real um/pixel calibration, which isn't
    available for any of datasets/DO|EH|VM (confirmed against the source
    dataset's GitHub repo, paper, and CVPR supplementary -- see PLAN.md
    Phase 3). Without this, `build_graph.py`'s node features are raw pixel
    values, and pooling them across images of very different native
    resolution (confirmed via src/models/gvae/diagnose_image_variance.py to
    explain ~86% of the GVAE latent space's between-image variance) mostly
    encodes which image's resolution a cell came from, not its real
    (relative) size.

    Deliberately normalizes by the tissue's own extent, not the raw image
    canvas (sqrt(H*W)/H*W, tried first -- see PLAN.md Phase 3): images vary
    in how tightly they're cropped around the specimen, so canvas-based
    normalization let cropping margin leak into radial_distance_norm's
    meaning and measurably weakened the radial-position signal (GVAE latent
    correlation with radial_distance dropped from -0.80 to -0.44). Uses the
    99th percentile rather than the true max so one stray mis-segmented
    cell far from center can't blow up the scale for the whole image.
    """
    table = regionprops_table(labels, properties=PROPS)
    df = pd.DataFrame(table)
    df = df.rename(columns={
        "centroid-0": "centroid_row",
        "centroid-1": "centroid_col",
        "equivalent_diameter_area": "equivalent_diameter",
    })
    df = df[(df["area"] >= min_area) & (df["minor_axis_length"] >= min_minor_axis)].reset_index(drop=True)

    if max_area_log_std is not None and len(df) > 1:
        log_area = np.log(df["area"])
        max_area = np.exp(log_area.mean() + max_area_log_std * log_area.std())
        df = df[df["area"] <= max_area].reset_index(drop=True)

    # circularity: 1.0 = perfect circle, lower = more irregular/elongated boundary
    df["circularity"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2).clip(lower=1e-6)

    # elongation: major/minor axis ratio, >=1, higher = more elongated
    df["elongation"] = df["major_axis_length"] / df["minor_axis_length"].clip(lower=1e-6)

    if len(df) == 0:
        # Empty after filtering (e.g. a small patch crop with no cells
        # clearing min_area/min_minor_axis) -- np.average below would raise
        # ZeroDivisionError on an empty weights array. Return the empty
        # frame with the rest of the expected columns present but empty,
        # rather than crashing every caller that might hand this a crop.
        for col in ["radial_distance", "angular_position", "radial_orientation_delta"]:
            df[col] = pd.Series(dtype=float)
        if um_per_pixel is not None:
            for col in _UM_COLS:
                df[col] = pd.Series(dtype=float)
        if normalize_by_image_size:
            for col in _NORM_COLS:
                df[col] = pd.Series(dtype=float)
        return df

    # radial coordinate system anchored on the whole-section centroid (area-weighted
    # mean of cell centroids is a reasonable proxy for the section center).
    section_row = np.average(df["centroid_row"], weights=df["area"])
    section_col = np.average(df["centroid_col"], weights=df["area"])
    d_row = df["centroid_row"] - section_row
    d_col = df["centroid_col"] - section_col
    df["radial_distance"] = np.hypot(d_row, d_col)
    df["angular_position"] = np.arctan2(d_row, d_col)

    # orientation relative to the local radial direction: 0 = cell's major axis
    # points toward/away from section center (radial file), pi/2 = tangential.
    radial_angle = np.arctan2(d_row, d_col)
    rel = df["orientation"] - radial_angle
    df["radial_orientation_delta"] = np.abs(np.arctan2(np.sin(rel), np.cos(rel)))

    if um_per_pixel is not None:
        df["area_um2"] = df["area"] * um_per_pixel**2
        df["perimeter_um"] = df["perimeter"] * um_per_pixel
        df["equivalent_diameter_um"] = df["equivalent_diameter"] * um_per_pixel
        df["major_axis_length_um"] = df["major_axis_length"] * um_per_pixel
        df["minor_axis_length_um"] = df["minor_axis_length"] * um_per_pixel
        df["radial_distance_um"] = df["radial_distance"] * um_per_pixel

    if normalize_by_image_size:
        # Normalize by the *tissue's* own extent, not the raw image canvas
        # (sqrt(H*W)) -- images vary in how tightly they're cropped around
        # the specimen (some with lots of surrounding white slide margin,
        # some tight), so two images of the same real specimen size but
        # different framing would otherwise get different normalization
        # scales purely from margin, corrupting radial_distance_norm's
        # meaning as "fraction of the way from pith to edge" in particular.
        # tissue_radius (99th percentile, not max, so one stray
        # far-flung mis-segmented cell can't blow up the scale for the
        # whole image) is derived from the cells' own layout instead, and
        # used consistently as the one characteristic length for every
        # size column so they all stay on the same relative scale.
        tissue_radius = float(np.percentile(df["radial_distance"], 99)) if len(df) > 1 else 1.0
        tissue_radius = max(tissue_radius, 1e-6)
        tissue_area = tissue_radius**2
        df["area_norm"] = df["area"] / tissue_area
        df["perimeter_norm"] = df["perimeter"] / tissue_radius
        df["equivalent_diameter_norm"] = df["equivalent_diameter"] / tissue_radius
        df["major_axis_length_norm"] = df["major_axis_length"] / tissue_radius
        df["minor_axis_length_norm"] = df["minor_axis_length"] / tissue_radius
        df["radial_distance_norm"] = df["radial_distance"] / tissue_radius

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if labels_path is None:
        raise SystemExit("usage: python -m src.graph.features <labels.npy>")
    labels = np.load(labels_path)
    df = extract_cell_features(labels)
    print(df.describe())
    print(f"{len(df)} cells")
