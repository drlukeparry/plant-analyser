"""R3 (ROADMAP2's naming) fast-preview geometry rendering, wired up for the
set-Transformer DDIM's output: a weighted (Laguerre/power-diagram) Voronoi
tessellation from generated centroids + sizes, so a generated cell's target
size actually determines its rendered polygon's size, not just its
centroid's spacing (per ROADMAP2 R3's explicit reason for preferring
weighted over plain Voronoi).

Implemented directly (Sutherland-Hodgman half-plane clipping, numpy only --
no new dependency) rather than pulling in a power-diagram library, matching
this repo's "start small" precedent (Phase 5's hand-rolled UNet rather than
a diffusion-model library). This is the "fast preview tier" ROADMAP2 R3
describes -- straight-edged/convex cells, not fitted wall curvature. The
higher-fidelity relaxation tier is explicitly out of scope here.

Power-diagram definition: cell_i = { p : |p - c_i|^2 - w_i <= |p - c_j|^2 - w_j
for all j != i }, which simplifies to a linear half-plane per (i, j) pair --
this is what lets it be built via repeated polygon clipping instead of a
generalized convex-hull-lifting computation.
"""
import numpy as np


def clip_polygon_halfplane(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Sutherland-Hodgman clip: keep the part of `poly` where normal @ p <= offset."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        curr, nxt = poly[i], poly[(i + 1) % n]
        curr_in = normal @ curr <= offset
        nxt_in = normal @ nxt <= offset
        if curr_in:
            out.append(curr)
        if curr_in != nxt_in:
            d = nxt - curr
            denom = normal @ d
            if abs(denom) < 1e-12:
                continue
            t = (offset - normal @ curr) / denom
            out.append(curr + t * d)
    return np.array(out) if out else np.zeros((0, 2))


def power_diagram(points: np.ndarray, weights: np.ndarray, margin_frac: float = 0.15
                   ) -> tuple[list[np.ndarray], np.ndarray]:
    """points: [N, 2], weights: [N] (>= 0, larger = bigger cell, w_i ~= radius_i^2).
    Returns (cells, box): cells is a list of N polygons (each an [M, 2]
    array of vertices, possibly empty if a low-weight site is fully
    absorbed by its neighbors -- a real property of power diagrams, not a
    bug); box is the clipping rectangle used, needed by callers to identify
    boundary cells (see `is_boundary_cell`) whose polygon is an arbitrary
    function of the clip box, not of the site's own weight."""
    n = len(points)
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    m = margin_frac * max(xmax - xmin, ymax - ymin, 1e-6)
    box = np.array([
        [xmin - m, ymin - m], [xmax + m, ymin - m],
        [xmax + m, ymax + m], [xmin - m, ymax + m],
    ])

    lift = (points ** 2).sum(axis=1) - weights  # |p_i|^2 - w_i, reused per pair
    cells = []
    for i in range(n):
        poly = box.copy()
        for j in range(n):
            if j == i or len(poly) == 0:
                continue
            normal = 2 * (points[j] - points[i])
            if np.allclose(normal, 0):
                continue
            offset = lift[j] - lift[i]
            poly = clip_polygon_halfplane(poly, normal, offset)
        cells.append(poly)
    return cells, box


def is_boundary_cell(poly: np.ndarray, box: np.ndarray, tol: float = 1e-6) -> bool:
    """True if any vertex of `poly` lies on the clip box's edge -- such a
    cell's size/shape is an artifact of the arbitrary clip box, not a
    meaningful function of the site's own weight (it's only "large" because
    it extends unbounded toward the boundary), and should be excluded from
    any area-based comparison."""
    if len(poly) == 0:
        return False
    xmin, ymin = box.min(axis=0)
    xmax, ymax = box.max(axis=0)
    on_x = np.isclose(poly[:, 0], xmin, atol=tol) | np.isclose(poly[:, 0], xmax, atol=tol)
    on_y = np.isclose(poly[:, 1], ymin, atol=tol) | np.isclose(poly[:, 1], ymax, atol=tol)
    return bool(np.any(on_x | on_y))


def cell_positions_and_weights(attrs: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """attrs: [N, attr_dim] de-standardized attribute vectors (real or
    generated). Recovers (x, y) from radial_distance_norm/angular_position
    using the same (row, col) convention control_fields.py's
    `_normalized_xy` already established, and a power-diagram weight
    w = (equivalent_diameter_norm / 2)^2 (radius^2) from each cell's own
    generated/real size."""
    r = attrs[:, feature_names.index("radial_distance_norm")]
    theta = attrs[:, feature_names.index("angular_position")]
    d_row = r * np.sin(theta)
    d_col = r * np.cos(theta)
    points = np.stack([d_col, d_row], axis=1)
    diam = np.clip(attrs[:, feature_names.index("equivalent_diameter_norm")], 1e-6, None)
    weights = (diam / 2) ** 2
    return points, weights
