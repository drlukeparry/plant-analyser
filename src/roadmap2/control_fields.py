"""Extract real, per-cell "design control fields" (local size-gradient target,
local alignment-flow direction) from a real image's node_df, per ROADMAP2's
"Extracting real control fields" section. These become (a) R2's per-node
training targets/conditioning inputs, and (b) a library of real fields that
can later be transplanted onto a novel boundary shape.

Both fields are computed leave-one-out (a node's own value never contributes
to its own target) via k-nearest-neighbor spatial smoothing in normalized
polar->cartesian coordinates (radial_distance_norm, angular_position -- the
same tissue-extent-normalized coordinate system build_graph.py/features.py
already established, so this is comparable across images of different
resolution/scale without recomputing anything from raw pixels).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _normalized_xy(node_df: pd.DataFrame) -> np.ndarray:
    r = node_df["radial_distance_norm"].values
    theta = node_df["angular_position"].values
    # angular_position = arctan2(d_row, d_col) in features.py -- keep the same
    # (row, col) axis convention so this stays consistent with the rest of
    # the codebase, even though for KNN purposes any consistent 2D embedding works.
    d_row = r * np.sin(theta)
    d_col = r * np.cos(theta)
    return np.stack([d_row, d_col], axis=1)


def size_gradient_field(node_df: pd.DataFrame, k: int = 8) -> np.ndarray:
    """Leave-one-out local mean of equivalent_diameter_norm over each node's
    k nearest neighbors (in normalized tissue coordinates) -- a smooth
    per-node "what size should a cell here be" target."""
    xy = _normalized_xy(node_df)
    sizes = node_df["equivalent_diameter_norm"].values
    n = len(node_df)
    k_eff = min(k + 1, n)  # +1 since the node itself is always its own nearest neighbor
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff)
    if k_eff == 1:
        return sizes.copy()
    neighbor_idx = idx[:, 1:]  # drop self (index 0 after query, always closest)
    return sizes[neighbor_idx].mean(axis=1)


def alignment_flow_field(node_df: pd.DataFrame, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out local circular mean of `orientation` over each node's k
    nearest neighbors. `orientation` (skimage regionprops convention) is
    axial/180-degree-periodic, not a full 360-degree direction -- e.g. a cell
    at +85 degrees and one at -85 degrees are nearly the same alignment, not
    opposite. Handled via the standard doubled-angle circular-mean trick:
    average sin(2*theta)/cos(2*theta), then halve the resulting angle back.
    Returns (flow_angle, flow_coherence) -- coherence in [0, 1] is the
    resultant vector length, i.e. how consistent the local alignment is (1 =
    all neighbors point the same way, 0 = directions cancel out / random)."""
    xy = _normalized_xy(node_df)
    theta = node_df["orientation"].values
    n = len(node_df)
    k_eff = min(k + 1, n)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff)
    if k_eff == 1:
        return theta.copy(), np.ones(n)
    neighbor_idx = idx[:, 1:]
    neighbor_theta = theta[neighbor_idx]
    sin2 = np.sin(2 * neighbor_theta).mean(axis=1)
    cos2 = np.cos(2 * neighbor_theta).mean(axis=1)
    flow_angle = 0.5 * np.arctan2(sin2, cos2)
    flow_coherence = np.hypot(sin2, cos2)
    return flow_angle, flow_coherence


def add_control_fields(node_df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Returns a copy of node_df with control-field columns added:
    size_gradient_target, alignment_flow_angle, alignment_flow_coherence."""
    out = node_df.copy()
    out["size_gradient_target"] = size_gradient_field(node_df, k=k)
    flow_angle, flow_coherence = alignment_flow_field(node_df, k=k)
    out["alignment_flow_angle"] = flow_angle
    out["alignment_flow_coherence"] = flow_coherence
    return out
