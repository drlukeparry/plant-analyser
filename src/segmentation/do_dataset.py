"""Loader for datasets/DO (potato tuber images + tissue-region masks).

Note: the DO annotation masks are semantic tissue-region labels (~9 classes:
-1 background, 0..8 tissue regions such as periderm/cortex/vascular ring/pith),
not per-cell instance masks. They are useful as auxiliary tissue-type ground
truth (Phase 2/3), not as instance-segmentation training targets for Cellpose.
"""
from dataclasses import dataclass
from pathlib import Path

DO_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "DO"

# No scale/calibration metadata ships with datasets/DO (no EXIF on the JPGs,
# TIFF annotations carry XResolution=YResolution=(1,1) i.e. uncalibrated, and
# no accompanying metadata files). Derived instead from the source paper
# (Biswas & Barma, Scientific Data 2020, DOI 10.1038/s41597-020-00706-9),
# which states images were captured at fixed 3x zoom with a field of view of
# 890 x 740 um^2. That FOV, divided by this specific file's actual pixel
# dimensions (5489x4522 for DO_0000.jpg, not the ~3650x3000 the paper's own
# text associates with 0.26 um/px), gives two independent estimates that
# agree to ~1% (890/5489=0.1622, 740/4522=0.1636 um/px) -- decent evidence
# the physical FOV-per-image assumption is right and these files are just a
# higher-resolution digitization/stitch than the paper's own example.
# Pixel dimensions vary a lot across datasets/DO (checked: from ~2964x2723 to
# 8786x9815), which is consistent with per-image stitched panoramas sized to
# each tuber's actual extent at a constant um/pixel, not a per-image zoom
# change -- so this constant is applied dataset-wide, not just to DO_0000.
# Treat as a documented working assumption, not verified ground truth --
# override this if a more authoritative per-image calibration turns up.
DO_UM_PER_PIXEL = 0.1629

# CAVEAT (unresolved as of Phase 2): applying this constant to the current
# classical_watershed segmentation gives a median cell equivalent diameter of
# ~1.1 um -- implausible for potato tuber parenchyma, which is well documented
# as unusually large (~100-170 um), among the largest plant cells; periderm
# cork cells are smaller but still ~15-40 um. The raw-pixel median equivalent
# diameter is only ~8px regardless of scale choice, which is too large a gap
# (~2 orders of magnitude) to be calibration error alone. Likely cause:
# watershed over-segmentation -- the source paper explicitly notes ground-truth
# generation required morphological cleanup to remove starch-granule artifacts
# (internal cell contents create false internal wall-like edges), which
# classical_watershed.py does not yet do. Treat both DO_UM_PER_PIXEL and any
# absolute size figures derived from it as provisional until this is resolved
# (needs either starch-granule-aware cleanup in segmentation, or an independent
# in-image scale reference to decouple the two open questions).


@dataclass(frozen=True)
class DOSample:
    name: str
    image_path: Path
    annotation_path: Path


def _read_split(split_images_txt: Path, split_annotations_txt: Path) -> list[DOSample]:
    image_lines = split_images_txt.read_text().splitlines()
    ann_lines = split_annotations_txt.read_text().splitlines()
    if len(image_lines) != len(ann_lines):
        raise ValueError(
            f"{split_images_txt.name} has {len(image_lines)} entries but "
            f"{split_annotations_txt.name} has {len(ann_lines)}"
        )
    samples = []
    for img_rel, ann_rel in zip(image_lines, ann_lines):
        img_path = DO_ROOT / img_rel
        ann_path = DO_ROOT / ann_rel
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        if not ann_path.exists():
            raise FileNotFoundError(ann_path)
        samples.append(DOSample(name=img_path.stem, image_path=img_path, annotation_path=ann_path))
    return samples


def load_train_split() -> list[DOSample]:
    return _read_split(DO_ROOT / "train_inputimages.txt", DO_ROOT / "train_annotations.txt")


def load_test_split() -> list[DOSample]:
    test_images_txt = DO_ROOT / "test_inputimages.txt"
    test_ann_txt = DO_ROOT / "test_annotations.txt"
    if test_ann_txt.exists():
        return _read_split(test_images_txt, test_ann_txt)
    # No test_annotations.txt shipped; derive annotation paths by filename convention.
    image_lines = test_images_txt.read_text().splitlines()
    samples = []
    for img_rel in image_lines:
        img_path = DO_ROOT / img_rel
        ann_path = DO_ROOT / "annotations" / f"{img_path.stem}.tiff"
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        if not ann_path.exists():
            raise FileNotFoundError(ann_path)
        samples.append(DOSample(name=img_path.stem, image_path=img_path, annotation_path=ann_path))
    return samples


if __name__ == "__main__":
    train = load_train_split()
    test = load_test_split()
    print(f"train: {len(train)} samples, test: {len(test)} samples")
    print("first train sample:", train[0])
