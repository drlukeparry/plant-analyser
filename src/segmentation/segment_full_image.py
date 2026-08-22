"""Run segment_classical on one full-resolution datasets/DO image and save the
resulting instance label map, mirroring how outputs/phase1/DO_0000_full_labels.npy
was produced. Usage: uv run python -m src.segmentation.segment_full_image DO_0022
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.classical_watershed import segment_classical
from src.segmentation.do_dataset import DO_ROOT

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "DO_0022"
    image_path = DO_ROOT / "inputimages" / f"{name}.jpg"
    rgb = np.array(Image.open(image_path).convert("RGB"))
    print(f"segmenting {name} ({rgb.shape[1]}x{rgb.shape[0]}) ...")

    t0 = time.time()
    labels, _ = segment_classical(rgb)
    print(f"done in {time.time()-t0:.1f}s: {labels.max()} cells")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}_full_labels.npy"
    np.save(out_path, labels)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
