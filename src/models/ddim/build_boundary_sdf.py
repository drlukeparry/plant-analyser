"""Precomputes the whole-silhouette `boundary_sdf` channel (field.py) for
each DO image, at native resolution, cached to disk once -- the SDF-crop
conditioning experiment needs this available before build_patches.py can
crop windows out of it per-patch.

Separate from build_patches.py because this is a per-image, one-time cost
(~10s for the largest DO image, per direct timing) that shouldn't be
recomputed on every patch-sampling run.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.field import rasterize_boundary_sdf

LABELS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase1"
DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5" / "boundary_sdf"


def main(species: str = "DO", tag: str = "norm"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graph_paths = sorted(out_dir_for(tag).glob(f"{species}_*.pt"))
    print(f"{len(graph_paths)} {species} images")

    t0 = time.time()
    for i, gp in enumerate(graph_paths):
        name = gp.stem
        out_path = OUT_DIR / f"{name}.npy"
        if out_path.exists():
            continue

        import torch
        cached = torch.load(gp, weights_only=False)
        node_df = cached["node_df"]
        tissue_radius = max(float(np.percentile(node_df["radial_distance"], 99)), 1e-6) \
            if len(node_df) > 1 else 1.0

        labels = np.load(LABELS_DIR / f"{name}_full_labels.npy")
        rgb = np.array(Image.open(DATASETS_DIR / species / "inputimages" / f"{name}.jpg").convert("RGB"))
        assert rgb.shape[:2] == labels.shape, f"{name}: rgb {rgb.shape[:2]} != labels {labels.shape}"

        t_img = time.time()
        boundary_sdf = rasterize_boundary_sdf(rgb, tissue_radius=tissue_radius)
        np.save(out_path, boundary_sdf)
        print(f"[{i+1}/{len(graph_paths)}] {name}: {labels.shape}, "
              f"tissue_radius={tissue_radius:.1f}px, {time.time()-t_img:.1f}s")

    print(f"done in {time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    species = sys.argv[1] if len(sys.argv) > 1 else "DO"
    main(species)
