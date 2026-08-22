"""Zero-shot Cellpose-SAM baseline on one DO sample, with a QC overlay.

datasets/DO annotations are tissue-region (not per-cell instance) masks, so
there is no instance ground truth to fine-tune against here. This script
establishes the zero-shot baseline and overlays the tissue-region mask so
per-cell instances can later be tagged with a tissue-type label (Phase 2/3).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.segmentation.do_dataset import load_train_split

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"
MAX_SIDE = 1024  # downsample for a fast prototype run


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_and_downsample(image_path: Path, max_side: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    return np.array(img)


def load_and_downsample_labels(ann_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    labels = tifffile.imread(ann_path)
    im = Image.fromarray(labels.astype(np.int32), mode="I")
    im = im.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(im)


def main():
    from cellpose import models

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_train_split()[0]
    print(f"Running zero-shot Cellpose-SAM on {sample.name} (device={device()})")

    img = load_and_downsample(sample.image_path, MAX_SIDE)
    tissue = load_and_downsample_labels(sample.annotation_path, img.shape[:2])

    model = models.CellposeModel(gpu=True, device=device())
    masks, flows, styles = model.eval(img, channels=None, diameter=None)

    n_cells = masks.max()
    print(f"Detected {n_cells} candidate cell instances at {img.shape[:2]} resolution")

    np.save(OUT_DIR / f"{sample.name}_masks.npy", masks)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input (downsampled)")
    axes[1].imshow(masks, cmap="nipy_spectral")
    axes[1].set_title(f"Cellpose-SAM instances (n={n_cells})")
    axes[2].imshow(tissue, cmap="tab10")
    axes[2].set_title("DO tissue-region labels (semantic, not instance)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{sample.name} — Phase 1 zero-shot baseline")
    fig.tight_layout()
    out_path = OUT_DIR / f"{sample.name}_baseline_qc.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved QC figure to {out_path}")


if __name__ == "__main__":
    main()
