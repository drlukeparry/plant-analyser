"""Batch-run segment_classical over every image in a given datasets/<NAME>
dataset and save each label map to outputs/phase1/<image_name>_full_labels.npy,
skipping images already segmented. Logs progress/timing/failures per image so
a long batch run is inspectable without re-running everything.

Usage: uv run python -m src.segmentation.segment_all [DATASET_NAME]
(defaults to DO; DATASET_NAME must match a datasets/<DATASET_NAME>/inputimages/ dir)
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# datasets/VM images exceed PIL's default decompression-bomb pixel-count
# guard (~10211x9889 = ~101M px vs. the ~89M default limit) -- these are
# known-legitimate high-res micrographs, not a DOS payload.
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.classical_watershed import segment_classical

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"
LOG_PATH = OUT_DIR / "segment_all_log.txt"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "DO"
    images_dir = DATASETS_ROOT / dataset_name / "inputimages"
    if not images_dir.exists():
        raise FileNotFoundError(f"no such dataset dir: {images_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob(f"{dataset_name}_*.jpg"))
    log(f"\n=== segment_all run start ({dataset_name}), {len(image_paths)} images found ===")

    done, skipped, failed = 0, 0, 0
    for i, image_path in enumerate(image_paths):
        name = image_path.stem
        out_path = OUT_DIR / f"{name}_full_labels.npy"
        if out_path.exists():
            skipped += 1
            log(f"[{i+1}/{len(image_paths)}] {name}: already done, skipping")
            continue

        try:
            rgb = np.array(Image.open(image_path).convert("RGB"))
            t0 = time.time()
            labels, _ = segment_classical(rgb)
            dt = time.time() - t0
            np.save(out_path, labels)
            done += 1
            log(f"[{i+1}/{len(image_paths)}] {name}: {rgb.shape[1]}x{rgb.shape[0]}, "
                f"{labels.max()} cells, {dt:.1f}s -> {out_path.name}")
        except Exception as e:
            failed += 1
            log(f"[{i+1}/{len(image_paths)}] {name}: FAILED - {e}")
            log(traceback.format_exc())

    log(f"=== segment_all run complete: {done} newly segmented, {skipped} skipped, "
        f"{failed} failed (of {len(image_paths)} total) ===")


if __name__ == "__main__":
    main()
