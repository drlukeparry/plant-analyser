"""Phase 3 follow-up check: is the remaining num_cells/megapixels correlation
found in diagnose_image_variance.py (even after the v2 tissue-extent
normalization fix) a segment_classical resolution artifact, or just real
specimens being proportionally larger? Computes cell density per unit
*tissue* area (not per unit image area) and correlates it against image
resolution directly -- if segmentation behavior were resolution-sensitive,
density-per-tissue-area should still track resolution; if it's just larger
specimens digitized at correspondingly higher resolution, it shouldn't.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_all_graphs import out_dir_for

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"


def main(variant: str = "norm"):
    graph_paths = sorted(out_dir_for(variant).glob("*.pt"))
    rows = []
    for p in graph_paths:
        cached = torch.load(p, weights_only=False)
        node_df = cached["node_df"]
        name = p.stem
        code = name.split("_")[0]
        img_path = DATASETS_ROOT / code / "inputimages" / f"{name}.jpg"
        w, h = Image.open(img_path).size
        tissue_radius = float(np.percentile(node_df["radial_distance"], 99))
        rows.append({
            "name": name,
            "species": code,
            "megapixels": w * h / 1e6,
            "num_cells": len(node_df),
            "tissue_radius": tissue_radius,
            "cells_per_tissue_area": len(node_df) / tissue_radius**2,
        })
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "cell_density_vs_resolution.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")

    print(f"\ncorr(cells_per_tissue_area, megapixels) pooled = "
          f"{np.corrcoef(df['cells_per_tissue_area'], df['megapixels'])[0, 1]:+.3f}")
    print(f"corr(tissue_radius, megapixels)         pooled = "
          f"{np.corrcoef(df['tissue_radius'], df['megapixels'])[0, 1]:+.3f}")
    print(f"corr(num_cells, megapixels)             pooled = "
          f"{np.corrcoef(df['num_cells'], df['megapixels'])[0, 1]:+.3f}")
    for code in sorted(df["species"].unique()):
        sub = df[df["species"] == code]
        r = np.corrcoef(sub["cells_per_tissue_area"], sub["megapixels"])[0, 1]
        print(f"  {code}: corr(cells_per_tissue_area, megapixels) = {r:+.3f} (n={len(sub)})")

    return df


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "norm"
    main(variant)
