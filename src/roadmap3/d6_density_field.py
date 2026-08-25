"""D6 prototype: rasterize a real image's cell positions into a small,
fixed-resolution (boundary_mask, density) grid pair -- the training
data/target for a scale-invariant placement model, per ROADMAP3.md's D6
scoping. Unlike Phase4/5's `field.py` (which rasterizes from a native-
resolution label image), this rasterizes directly from node_df's existing
tissue-normalized coordinates (`radial_distance_norm`/`angular_position`,
the same "_norm" system `control_fields.py` already uses) onto a small,
CONSISTENT grid across all images -- no raw label images needed, and no
per-image resolution/scale leak (the exact problem Phase 3 already found
and fixed for node features, here avoided from the start by construction).

Grid covers a fixed [-GRID_EXTENT, GRID_EXTENT]^2 window in normalized
coordinates for every image -- chosen from the real dataset's own
radial_distance_norm range (tissue-extent-normalized, so ~1.0-1.2 covers
the typical full tissue radius; see full_structure.py's own measurement of
DO_0000's range, 0.01-1.12).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from src.roadmap2.control_fields import _normalized_xy, local_density_field

GRID_EXTENT = 1.3
GRID_RES = 48


def _grid_coords(res: int = GRID_RES, extent: float = GRID_EXTENT):
    lin = np.linspace(-extent, extent, res)
    gx, gy = np.meshgrid(lin, lin)
    return gx, gy, lin


def rasterize_image(node_df, res: int = GRID_RES, extent: float = GRID_EXTENT,
                     boundary_k: int = 3, smooth_sigma: float = 1.2):
    """Returns (boundary_mask, density_raster), each (res, res) float32.

    boundary_mask: 1.0 where the grid point is "inside tissue" (close to
    real cells), 0.0 outside -- estimated as within `boundary_k`-NN mean
    distance of the real point set, a cheap silhouette proxy that needs no
    separate boundary/mask data.
    density_raster: local cell density (from `local_density_field`, cells
    per unit normalized area) scattered onto the nearest grid cell then
    Gaussian-smoothed, matching this codebase's existing paint-then-smooth
    rasterization pattern (`field.py`'s `paint_by_feature` + smoothing
    precedent) -- only defined (nonzero) inside boundary_mask.
    """
    xy = _normalized_xy(node_df)
    density = local_density_field(node_df)
    gx, gy, lin = _grid_coords(res, extent)
    grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)

    tree = cKDTree(xy)
    k_eff = min(boundary_k, len(xy))
    d_to_real, _ = tree.query(grid_xy, k=k_eff)
    if k_eff > 1:
        d_to_real = d_to_real.mean(axis=1)
    typical_spacing = np.median(tree.query(xy, k=min(2, len(xy)))[0][:, -1]) if len(xy) > 1 else 0.05
    boundary_mask = (d_to_real < 2.5 * typical_spacing).astype(np.float32).reshape(res, res)

    # scatter each real cell's own density value onto its nearest grid cell
    cell_size = (2 * extent) / (res - 1)
    ix = np.clip(np.round((xy[:, 0] + extent) / cell_size).astype(int), 0, res - 1)
    iy = np.clip(np.round((xy[:, 1] + extent) / cell_size).astype(int), 0, res - 1)
    accum = np.zeros((res, res), dtype=np.float64)
    counts = np.zeros((res, res), dtype=np.float64)
    np.add.at(accum, (iy, ix), density)
    np.add.at(counts, (iy, ix), 1.0)
    raw = np.divide(accum, counts, out=np.zeros_like(accum), where=counts > 0)
    density_raster = gaussian_filter(raw, sigma=smooth_sigma).astype(np.float32)
    density_raster *= boundary_mask  # zero outside tissue, matches Phase4/5's "only meaningful inside" convention

    return boundary_mask, density_raster


def density_at(density_raster: np.ndarray, xy: np.ndarray, res: int = GRID_RES,
               extent: float = GRID_EXTENT) -> np.ndarray:
    """Bilinear lookup of `density_raster` at world-coordinate points `xy`
    ([N, 2]). Used at sampling time to query the predicted density field at
    an arbitrary candidate point, not just grid nodes."""
    cell_size = (2 * extent) / (res - 1)
    fx = (xy[:, 0] + extent) / cell_size
    fy = (xy[:, 1] + extent) / cell_size
    fx = np.clip(fx, 0, res - 1 - 1e-6)
    fy = np.clip(fy, 0, res - 1 - 1e-6)
    x0, y0 = fx.astype(int), fy.astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = fx - x0, fy - y0
    v00 = density_raster[y0, x0]
    v10 = density_raster[y0, x1]
    v01 = density_raster[y1, x0]
    v11 = density_raster[y1, x1]
    return (v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy) +
            v01 * (1 - wx) * wy + v11 * wx * wy)


def density_to_spacing_diam(density: np.ndarray) -> np.ndarray:
    """Inverts the k-NN density estimator's area formula (density = k /
    (pi r_k^2), see `local_density_field`) to get a typical nearest-
    neighbor spacing for a given density -- the "diam" `seed_positions_
    blue_noise` expects as its per-point target spacing."""
    density = np.clip(density, 1e-6, None)
    return 2.0 * np.sqrt(1.0 / (np.pi * density))
