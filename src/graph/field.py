"""Phase 4: rasterize a cell-instance label image + its feature table into a
multi-channel field (signed-distance-to-wall, orientation, size, radial
distance) — the bridge representation a DDIM would operate on later.
"""
import numpy as np
from scipy import ndimage as ndi

from src.graph.viz import paint_by_feature

CHANNELS = ["sdf", "orientation", "equivalent_diameter", "radial_distance"]


def rasterize_field(labels: np.ndarray, node_df, bg_clip: float = 50.0) -> np.ndarray:
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
    """
    interior = labels > 0
    dist_in = ndi.distance_transform_edt(interior)
    dist_out = np.clip(ndi.distance_transform_edt(~interior), 0, bg_clip)
    sdf = np.where(interior, dist_in, -dist_out).astype(np.float32)

    orientation = paint_by_feature(labels, node_df, "orientation").astype(np.float32)
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
