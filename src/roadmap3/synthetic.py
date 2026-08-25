"""D4a + D5: synthetic boundary + explicit, novel control fields -- not
lifted from any real image, the actual point of the "arbitrary shape,
explicit controls" target (ROADMAP2's Final Target Specification, and
ROADMAP3.md's D5). Deliberately uses a steeper size gradient than typical
real images (per D5's own suggestion: "a gradient steeper than anything in
the real dataset" is the real test of generalization, not a gentle
in-distribution one).

Cell positions are NOT part of what the DDIM generates here -- per D4a,
they're seeded by a simple blue-noise-style point process (dart-throwing
rejection sampling against a local target-size-derived minimum spacing),
a deterministic step, same spirit as R3's "geometry is a deterministic
rendering step, not something the model learns implicitly." Only the
remaining 11 attribute dims (everything except radial_distance_norm/
angular_position) are actually diffused -- see inpaint.py.
"""
import numpy as np
from scipy.spatial import cKDTree


def circular_boundary(center: np.ndarray, radius: float):
    def inside(p: np.ndarray) -> bool:
        return np.linalg.norm(p - center) <= radius
    return inside


def radial_size_field(center: np.ndarray, radius: float, diam_center: float, diam_edge: float):
    """Target equivalent_diameter_norm as a function of position: larger
    near `center`, smaller near the boundary -- generalizes the single
    real-image radial trend control_fields.py extracts, but here it's an
    explicit, freely chosen input rather than something read off one image."""
    def size_at(p: np.ndarray) -> float:
        r = np.linalg.norm(p - center) / radius
        r = np.clip(r, 0.0, 1.0)
        return diam_center + (diam_edge - diam_center) * r
    return size_at


def tangential_flow_field(center: np.ndarray, coherence: float = 0.9):
    """A coherent swirl: alignment tangent to circles around `center`.
    Reduced mod pi (the same halved-double-angle convention
    control_fields.py's real alignment_flow_angle already uses, since
    orientation is 180-degree periodic) so it's directly comparable/
    interchangeable with the real per-node control field."""
    def flow_at(p: np.ndarray) -> tuple[float, float]:
        d = p - center
        raw_angle = np.arctan2(d[1], d[0]) + np.pi / 2
        angle = 0.5 * np.arctan2(np.sin(2 * raw_angle), np.cos(2 * raw_angle))
        return angle, coherence
    return flow_at


def seed_positions_blue_noise(inside_fn, size_field_fn, bbox: tuple[float, float, float, float],
                               spacing_factor: float = 3.5, max_candidates: int = 20000,
                               max_points: int | None = 100, seed: int = 0) -> np.ndarray:
    """Dart-throwing rejection sampling: draw random candidates in `bbox`,
    accept if inside the boundary and at least `spacing_factor *
    (own_target_diameter + nearest_accepted_diameter)/2` away from every
    already-accepted point -- denser where the size field is smaller. Not a
    rigorous Poisson-disk implementation (no guaranteed maximal packing),
    but sufficient for a fast-preview infill test.

    `spacing_factor` is not `1.0` (touching-diameter spacing) because real
    cells are irregular polygons packed by their actual (non-circular)
    shape, not circles at their diameter -- calibrated empirically (3.5)
    against real per-patch cell density (a n=41-cell real patch spanning
    ~0.028 units^2, per D3.5's render, is ~1500 cells/unit^2; `1.0` gives
    ~20,000 cells/unit^2, over an order of magnitude too dense). `max_points`
    caps the result at a size close to what the model was actually trained
    on (n_max in dataset.py) -- generating far more tokens than that is
    untested territory this pass doesn't attempt."""
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bbox
    accepted_pts, accepted_diam = [], []
    for _ in range(max_candidates):
        if max_points is not None and len(accepted_pts) >= max_points:
            break
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if not inside_fn(p):
            continue
        diam = size_field_fn(p)
        ok = True
        if accepted_pts:
            tree = cKDTree(np.array(accepted_pts)) if len(accepted_pts) > 20 else None
            if tree is not None:
                d, idx = tree.query(p, k=1)
                min_dist = spacing_factor * (diam + accepted_diam[idx]) / 2
                ok = d >= min_dist
            else:
                for q, dq in zip(accepted_pts, accepted_diam):
                    if np.linalg.norm(p - q) < spacing_factor * (diam + dq) / 2:
                        ok = False
                        break
        if ok:
            accepted_pts.append(p)
            accepted_diam.append(diam)
    return np.array(accepted_pts)
