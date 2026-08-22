"""Region-adjacency graph construction from an instance label image (Phase 2)."""
import numpy as np
import pandas as pd
import torch
from skimage.segmentation import expand_labels
from torch_geometric.data import Data

from src.graph.features import extract_cell_features

NODE_FEATURE_COLS_PX = [
    "area",
    "perimeter",
    "equivalent_diameter",
    "major_axis_length",
    "minor_axis_length",
    "elongation",
    "orientation",
    "solidity",
    "eccentricity",
    "circularity",
    "radial_distance",
    "angular_position",
    "radial_orientation_delta",
]

# When um_per_pixel is provided, size-related columns are swapped for their
# physical-unit equivalents so node features are comparable across images
# captured at different pixel resolutions (see datasets/DO pixel-dimension
# variability noted in src/segmentation/do_dataset.py). Ratios/angles are
# scale-invariant and unchanged.
_PX_TO_UM_COL = {
    "area": "area_um2",
    "perimeter": "perimeter_um",
    "equivalent_diameter": "equivalent_diameter_um",
    "major_axis_length": "major_axis_length_um",
    "minor_axis_length": "minor_axis_length_um",
    "radial_distance": "radial_distance_um",
}


def _pixel_adjacency_pairs(labels: np.ndarray, wall_expand_px: int = 22) -> pd.DataFrame:
    """Shared-wall pixel counts between touching label pairs, via 4-connected
    neighbor differencing (vectorized — avoids per-pair boundary tracing).

    Cell walls are several pixels thick, so interiors on either side don't
    touch directly in the raw label image — checking direct pixel adjacency
    alone massively undercounts real connectivity. Grow each label into the
    surrounding wall band first so neighboring cells meet at approximately
    the wall centerline before checking adjacency.

    `wall_expand_px` needs to be re-measured whenever the segmentation
    pipeline's wall detection changes materially, not treated as a fixed
    constant — the effective gap between labeled interiors is the true wall
    thickness *plus* whatever fraction of near-wall interior watershed
    leaves unlabeled, which varies with the segmenter. Measured directly:
    ~4px gave the right degree (~5.15) for an earlier segmentation version,
    but after the over-segmentation fix (local Otsu + h-maxima peak
    suppression) the effective gap measured ~16px, and `wall_expand_px=22`
    was needed to recover avg degree ~5 again.
    """
    expanded = expand_labels(labels, distance=wall_expand_px)
    pairs = []
    counts = []

    for a, b in [
        (expanded[:, :-1], expanded[:, 1:]),
        (expanded[:-1, :], expanded[1:, :]),
    ]:
        mask = (a > 0) & (b > 0) & (a != b)
        lo = np.minimum(a[mask], b[mask])
        hi = np.maximum(a[mask], b[mask])
        combined = np.stack([lo, hi], axis=1)
        uniq, cnt = np.unique(combined, axis=0, return_counts=True)
        pairs.append(uniq)
        counts.append(cnt)

    all_pairs = np.concatenate(pairs, axis=0)
    all_counts = np.concatenate(counts, axis=0)
    df = pd.DataFrame(all_pairs, columns=["label_a", "label_b"])
    df["shared_wall_px"] = all_counts
    df = df.groupby(["label_a", "label_b"], as_index=False)["shared_wall_px"].sum()
    return df


def build_graph(labels: np.ndarray, um_per_pixel: float | None = None) -> tuple[Data, pd.DataFrame]:
    node_df = extract_cell_features(labels, um_per_pixel=um_per_pixel)
    node_df = node_df.sort_values("label").reset_index(drop=True)
    label_to_idx = {lab: i for i, lab in enumerate(node_df["label"])}
    node_feature_cols = (
        [_PX_TO_UM_COL.get(c, c) for c in NODE_FEATURE_COLS_PX]
        if um_per_pixel is not None
        else NODE_FEATURE_COLS_PX
    )

    adj_df = _pixel_adjacency_pairs(labels)

    src, dst, shared_wall, centroid_dist, orient_delta, radial_alignment = [], [], [], [], [], []
    row_by_label = node_df.set_index("label")

    for _, r in adj_df.iterrows():
        a, b = int(r.label_a), int(r.label_b)
        if a not in label_to_idx or b not in label_to_idx:
            continue
        ia, ib = label_to_idx[a], label_to_idx[b]
        ra, rb = row_by_label.loc[a], row_by_label.loc[b]

        d_row = rb["centroid_row"] - ra["centroid_row"]
        d_col = rb["centroid_col"] - ra["centroid_col"]
        dist = float(np.hypot(d_row, d_col))
        edge_angle = np.arctan2(d_row, d_col)

        # radial direction at the edge midpoint (using the pair's mean radial angle)
        mid_angular = (ra["angular_position"] + rb["angular_position"]) / 2
        rel = edge_angle - mid_angular
        alignment = float(np.abs(np.cos(np.arctan2(np.sin(rel), np.cos(rel)))))  # 1=radial, 0=tangential

        rel_orient = ra["orientation"] - rb["orientation"]
        odelta = float(np.abs(np.arctan2(np.sin(rel_orient), np.cos(rel_orient))))

        for u, v in [(ia, ib), (ib, ia)]:
            src.append(u)
            dst.append(v)
            shared_wall.append(float(r.shared_wall_px))
            centroid_dist.append(dist)
            orient_delta.append(odelta)
            radial_alignment.append(alignment)

    x = torch.tensor(node_df[node_feature_cols].values, dtype=torch.float32)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    shared_wall_arr = np.array(shared_wall)
    centroid_dist_arr = np.array(centroid_dist)
    if um_per_pixel is not None:
        shared_wall_arr = shared_wall_arr * um_per_pixel  # pixel-count proxy -> approx um
        centroid_dist_arr = centroid_dist_arr * um_per_pixel
    edge_attr = torch.tensor(
        np.stack([shared_wall_arr, centroid_dist_arr, orient_delta, radial_alignment], axis=1),
        dtype=torch.float32,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_labels = torch.tensor(node_df["label"].values, dtype=torch.long)
    return data, node_df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.segmentation.do_dataset import DO_UM_PER_PIXEL

    labels = np.load(sys.argv[1])
    um_per_pixel = DO_UM_PER_PIXEL if "--um" in sys.argv else None
    data, node_df = build_graph(labels, um_per_pixel=um_per_pixel)
    print(data)
    print(f"nodes={data.num_nodes} edges={data.num_edges} (undirected pairs={data.num_edges // 2})")
    if um_per_pixel is not None:
        print(node_df[["area_um2", "equivalent_diameter_um", "major_axis_length_um"]].describe())
