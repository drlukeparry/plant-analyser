"""Phase 4: rasterize a cell-instance label image + its feature table into a
multi-channel field (signed-distance-to-wall, orientation, size, radial
distance) — the bridge representation a DDIM would operate on later.
"""
import numpy as np
from scipy import ndimage as ndi

from src.graph.viz import paint_by_feature

CHANNELS = ["sdf", "orientation", "equivalent_diameter", "radial_distance"]


def rasterize_field(labels: np.ndarray, node_df, bg_clip: float = 50.0,
                     normalize_by_size: bool = False) -> np.ndarray:
    """Returns a (H, W, 4) float32 field:
    - sdf: signed distance to nearest wall, positive inside a cell, negative
      in wall/background — a smooth representation of "cell-ness" a
      diffusion model can denoise toward, and from which seeds can be
      re-extracted (local maxima = cell centers).
    - orientation, equivalent_diameter, radial_distance: each cell's pixels
      filled with that cell's scalar value (broadcasts the graph's node
      features into image space).

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
        tissue_radius = max(float(np.percentile(node_df["radial_distance"], 99)), 1e-6) \
            if len(node_df) > 1 else 1.0
        sdf = sdf / tissue_radius
        size = paint_by_feature(labels, node_df, "equivalent_diameter_norm").astype(np.float32)
        radial = paint_by_feature(labels, node_df, "radial_distance_norm").astype(np.float32)
    else:
        size = paint_by_feature(labels, node_df, "equivalent_diameter").astype(np.float32)
        radial = paint_by_feature(labels, node_df, "radial_distance").astype(np.float32)

    return np.stack([sdf, orientation, size, radial], axis=-1)


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
