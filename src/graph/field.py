"""Phase 4: rasterize a cell-instance label image + its feature table into a
multi-channel field (signed-distance-to-wall, orientation, size) -- the
bridge representation a DDIM would operate on later.
"""
import numpy as np
from scipy import ndimage as ndi

from src.graph.viz import paint_by_feature

CHANNELS = ["sdf", "orientation", "equivalent_diameter"]
# radial_distance (distance to a single tissue centroid) used to be a 4th
# target channel here, but it's a flawed proxy for position -- it assumes a
# single center and breaks down for non-circular/lobed tissue shapes.
# rasterize_boundary_sdf below (true whole-silhouette distance) replaces it
# as a conditioning input rather than something the model has to reproduce.


def rasterize_field(labels: np.ndarray, node_df, bg_clip: float = 50.0,
                     normalize_by_size: bool = False) -> np.ndarray:
    """Returns a (H, W, 3) float32 field:
    - sdf: signed distance to nearest wall, positive inside a cell, negative
      in wall/background — a smooth representation of "cell-ness" a
      diffusion model can denoise toward, and from which seeds can be
      re-extracted (local maxima = cell centers).
    - orientation, equivalent_diameter: each cell's pixels filled with that
      cell's scalar value (broadcasts the graph's node features into image
      space).

    `bg_clip` bounds the negative (background/wall) side of the SDF to
    at most `-bg_clip` pixels, rather than leaving it as raw unbounded
    distance-to-nearest-cell (which reaches into the hundreds or low
    thousands of pixels for background far from any tissue -- see
    PLAN.md Phase 4). Found via `validate_roundtrip.py`'s degraded round
    trip: an unbounded SDF's dynamic range is dominated by these distant
    background values, so a fixed noise fraction of the channel's *range*
    becomes enormous relative to the few-pixel precision that actually
    matters near a real cell wall -- near-wall pixels had their
    inside/outside classification flipped 39% of the time by degradation
    that was only 5% of the (unbounded) range. Clipping removes that
    distortion at its source: distinguishing "500px from a cell" from
    "1200px from a cell" was never useful for this representation's actual
    purpose (marking cell interior vs. not), so there's no real
    information lost, only an artificially inflated dynamic range.

    `normalize_by_size=True` requires `node_df` to already carry the
    `*_norm` columns (`extract_cell_features(..., normalize_by_image_size=True)`)
    and additionally divides the `sdf` channel's values by the same
    per-image tissue-radius scale, using `equivalent_diameter_norm`/
    `radial_distance_norm` instead of the raw-pixel columns for the other
    two size-related channels. Phase 3 found training across images of very
    different native resolution with raw-pixel size features mostly encodes
    which image a patch came from, not real relative size (see PLAN.md
    Phase 3 -- ~86% of the GVAE's between-image latent variance traced to
    this before the fix) -- Phase 5 trains across the same 213-image,
    multi-resolution set, so the field needs the equivalent fix at rasterize
    time or it would reintroduce the identical bug one layer downstream.
    `bg_clip` is still applied in raw pixel units first (it's about the SDF's
    dynamic range, not cross-image comparability), then the whole sdf array
    is divided by the tissue scale. This is purely a value rescaling, not a
    change to the array's spatial (pixel) dimensions, so `tessellate.py`'s
    peak-detection is unaffected (it already normalizes per-connected-
    component, not against an absolute scale).
    """
    interior = labels > 0
    dist_in = ndi.distance_transform_edt(interior)
    dist_out = np.clip(ndi.distance_transform_edt(~interior), 0, bg_clip)
    sdf = np.where(interior, dist_in, -dist_out).astype(np.float32)

    orientation = paint_by_feature(labels, node_df, "orientation").astype(np.float32)
    if normalize_by_size:
        # radial_distance itself is no longer painted as an output channel (see CHANNELS'
        # docstring note) -- still used here only as the per-image scale for sdf/size
        # normalization, which is a different role (a scalar, not a per-pixel target).
        tissue_radius = max(float(np.percentile(node_df["radial_distance"], 99)), 1e-6) \
            if len(node_df) > 1 else 1.0
        sdf = sdf / tissue_radius
        size = paint_by_feature(labels, node_df, "equivalent_diameter_norm").astype(np.float32)
    else:
        size = paint_by_feature(labels, node_df, "equivalent_diameter").astype(np.float32)

    return np.stack([sdf, orientation, size], axis=-1)


def rasterize_boundary_sdf(rgb: np.ndarray, tissue_radius: float = 1.0, bg_clip: float = 50.0,
                            mask: np.ndarray | None = None) -> np.ndarray:
    """Signed distance to the whole tissue silhouette's boundary -- not to be
    confused with `rasterize_field`'s per-cell-wall `sdf` channel. Positive
    inside the tissue, negative outside, using the same coarse holes-filled
    silhouette segmentation itself uses to exclude background
    (`classical_watershed.tissue_mask`). This is the true "distance to the
    edge of the whole structure" signal that `radial_distance` (distance to
    a single centroid) was only ever a proxy for.

    `bg_clip` bounds the negative (outside-tissue) side, mirroring
    `rasterize_field`'s treatment of the wall sdf's background side, for the
    same reason: unbounded distance far from any tissue is dominated by
    scale that's never useful. The positive (inside) side is left unclipped
    in raw pixel units before the `tissue_radius` division -- "how deep into
    the structure is this pixel" up to the structure's own extent is exactly
    what this channel exists to carry, and it naturally lands near 1.0 at
    the tissue's core once divided by `tissue_radius`.

    `tissue_radius` should be the same per-image scale `normalize_by_size`
    uses for the wall sdf (99th-percentile `radial_distance`), so a crop of
    this channel lands in the same comparable, cross-image units as the
    target field it's conditioning.

    `mask` lets a precomputed silhouette be passed in (segmentation's own
    tissue_mask call is not free on a multi-thousand-pixel image) rather
    than recomputing it here every call.
    """
    if mask is None:
        from src.segmentation.classical_watershed import tissue_mask as compute_tissue_mask
        mask = compute_tissue_mask(rgb)
    dist_in = ndi.distance_transform_edt(mask)
    dist_out = np.clip(ndi.distance_transform_edt(~mask), 0, bg_clip)
    sdf = np.where(mask, dist_in, -dist_out).astype(np.float32)
    return sdf / tissue_radius


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.graph.features import extract_cell_features

    labels = np.load(sys.argv[1])
    node_df = extract_cell_features(labels)
    field = rasterize_field(labels, node_df)
    print("field shape:", field.shape, "channels:", CHANNELS)
    for i, name in enumerate(CHANNELS):
        print(f"  {name}: min={field[...,i].min():.2f} max={field[...,i].max():.2f} mean={field[...,i].mean():.2f}")
