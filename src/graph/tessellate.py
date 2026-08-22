"""Phase 4 inverse: field -> seed points (local SDF maxima) -> watershed
tessellation -> recovered instance labels. Closes the round-trip loop
(labels -> field -> tessellate -> recovered labels) used to validate the
field representation before it's used as a DDIM training target.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima


def tessellate_from_field(field: np.ndarray, peak_h: float = 4.0, peak_min_distance: int = 8,
                           smooth_sigma: float = 2.0) -> np.ndarray:
    sdf = field[..., 0]
    interior = sdf > 0

    distance_smooth = ndi.gaussian_filter(np.clip(sdf, 0, None), sigma=smooth_sigma)
    hmax = h_maxima(distance_smooth, h=peak_h)
    coords = peak_local_max(distance_smooth, min_distance=peak_min_distance, labels=interior & (hmax > 0))
    peak_mask = np.zeros_like(sdf, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    return segmentation.watershed(-distance_smooth, markers, mask=interior)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.graph.features import extract_cell_features
    from src.graph.field import rasterize_field

    labels = np.load(sys.argv[1])
    node_df = extract_cell_features(labels)
    field = rasterize_field(labels, node_df)
    recovered = tessellate_from_field(field)
    print(f"original: {labels.max()} cells, recovered: {recovered.max()} cells")
