"""Segments a single image, for driving segment_all.py's per-image work as
parallel OS processes (e.g. `xargs -P`) instead of one sequential loop --
see slurm/phase1_segment.slurm, which requests up to 96 cores on Ada's
`shortq` but segment_all.py itself never used more than one. Same
skip-if-exists/output-path convention as segment_all.py, so both are safe to
mix/resume against the same outputs/phase1/ directory.

Usage: uv run python -m src.segmentation.segment_one <path/to/image.jpg>
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # see segment_all.py -- VM images exceed the default guard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.classical_watershed import segment_classical

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"


def main(image_path: Path):
    name = image_path.stem
    out_path = OUT_DIR / f"{name}_full_labels.npy"
    if out_path.exists():
        print(f"{name}: already done, skipping")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        t0 = time.time()
        labels, _ = segment_classical(rgb)
        dt = time.time() - t0
        np.save(out_path, labels)
        print(f"{name}: {rgb.shape[1]}x{rgb.shape[0]}, {labels.max()} cells, "
              f"{dt:.1f}s -> {out_path.name}")
    except Exception as e:
        print(f"{name}: FAILED - {e}", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main(Path(sys.argv[1]))
