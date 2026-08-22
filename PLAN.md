# Plant Micrograph Structural Analysis + Generative Framework — Implementation Plan

Pipeline: segmentation → structural graph → Graph-VAE representation → field-based conditional DDIM generation.

Reference data:
- `datasets/examples/` — 3 stained stem cross-section micrographs (safranin/fast-green and toluidine-blue stains), used for the radial-tissue-zonation design work in Phases 2–4.
- `datasets/DO/` — **potato tuber cell dataset, already annotated**: 64 images in `inputimages/` with matching per-image instance masks in `annotations/` (`.tiff`/`.png`), and a pre-defined `train_inputimages.txt`/`train_annotations.txt` (22 images) / `test_inputimages.txt` (42 images) split. This gives Phase 1 a real annotated training/eval set from day one — no need to hand-correct Cellpose output from scratch before a first fine-tune.

---

## Phase 0 — Project setup & data audit
**Goal:** working environment, data inventory, no modeling yet.

- Base interpreter: the existing **`py311`** Anaconda environment (Python 3.11) — activate with `conda activate py311`.
- Dependency management: **`uv`**, driven by a generated **`pyproject.toml`** (no raw `pip install`, no `requirements.txt`). Steps:
  1. `conda activate py311`
  2. Generate `pyproject.toml` at repo root declaring the project and its dependencies: `cellpose`, `scikit-image`, `shapely`, `opencv-python`, `torch`, `torch_geometric`, `networkx`, `diffusers` (for `DDIMScheduler`), `albumentations`, `torchvision`.
  3. `uv venv --python $(python -c 'import sys; print(sys.executable)')` (or `uv venv --python 3.11`) to create/pin the uv-managed venv against the `py311` interpreter, then `uv sync` to install from `pyproject.toml`/lockfile.
  4. Use `uv run <script>` (or `uv run python -m ...`) for all subsequent phase commands instead of bare `python`, so the locked environment is always what executes.
- **Apple Silicon (MPS) backend:** all `torch` code should select device via
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
  rather than hardcoding `cuda`/`cpu`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the env for ops not yet implemented on MPS (some `torch_geometric` scatter/gather kernels may hit this — verify during Phase 3 setup and fall back to CPU per-op if needed).
- Repo structure:
  ```
  plant-analyser/
    pyproject.toml
    uv.lock
    datasets/
      examples/          # stem cross-section micrographs (radial zonation reference)
      DO/                # annotated potato tuber dataset (inputimages/, annotations/, train/test splits)
    src/segmentation/
    src/graph/
    src/models/gvae/
    src/models/ddim/
    src/eval/
    notebooks/            # exploratory only, not pipeline logic
  ```
- Inventory available images, stains, species, magnification/scale bars (needed for real-unit cell size later, e.g. the 200 µm scale bar visible in the `datasets/examples/` samples; check whether `datasets/DO/` has an equivalent scale reference).
- **Exit criteria:** `uv sync` completes inside `py311`; `torch.backends.mps.is_available()` returns `True`; both `datasets/examples/` and a sample from `datasets/DO/inputimages/` load and tile without errors.

---

## Phase 1 — Cell instance segmentation
**Goal:** reliable per-cell masks on both the annotated potato set and the stem cross-section stains.

- **Use `datasets/DO/` as the first fine-tuning target** — it already has 64 images with matching instance masks and a train (22) / test (42) split defined in `train_inputimages.txt`/`train_annotations.txt`/`test_inputimages.txt`. Write a loader that resolves these relative paths against `datasets/DO/` and confirms every listed image has a matching annotation.
- Fine-tune Cellpose (`cyto3`) directly on the `datasets/DO` train split (no manual correction needed for this set); evaluate on the held-out test split for real held-out metrics (mask IoU / AP) instead of just visual QC.
- Separately, run the resulting model zero-shot on tiled crops from `datasets/examples/` (the stem cross-sections) to check cross-domain transfer; expect it to struggle since potato tuber cells and stem cross-section cell walls differ in stain and morphology.
- Color-deconvolve each stem-section stain to isolate the wall channel before segmentation (Macenko or manual channel picking per stain).
- Hand-correct Cellpose output on 5–15 tiles from `datasets/examples/` (~200–500 cells/tile) using the Cellpose GUI, then fine-tune again (potato weights as the starting checkpoint) to adapt to the stem-section domain.
- Build a stitching step to merge tile-level masks back into whole-image instance maps (handle overlap-region duplicate instances).
- **Deliverables:** `src/segmentation/`, fine-tuned model checkpoint(s) (potato + stem-section variants), per-image instance mask (`.npy`/`.tif`) + polygon export (GeoJSON or shapely pickle).
- **Exit criteria:** quantitative IoU/AP on the `datasets/DO` test split meets a defined threshold; visual QC on `datasets/examples/` shows correct wall-following boundaries with no systematic merge/split errors in vascular ring or pith regions.

**Known issue found during zero-shot baseline (`DO_0000`):** Cellpose-SAM zero-shot correctly segments the inner parenchyma fan (cortex/vascular/pith) but detects almost nothing in the outer periderm band — confirmed via a native-resolution crop test, so it is not a downsampling artifact. The periderm's small, densely packed, uniformly-saturated cells with low wall contrast appear to collapse Cellpose's flow field rather than being individually resolved, regardless of pixel size.

Tested mitigations:
- ~~Explicit smaller `diameter` (tried `diameter=8` vs. auto)~~ — **ruled out**: periderm coverage was unchanged (5.9% vs 6.3% nonzero-pixel fraction in the periderm strip), and it introduced new spurious thin sliver false positives along radial walls elsewhere in the tissue, degrading the region that already worked. A single global diameter/threshold can't serve both tissue types.
- ~~Fine-tuning Cellpose on periderm-heavy tiles~~ — **superseded**, see below.

**Resolved — pivoted to classical image processing.** The `datasets/DO` paper's own ground-truth generation method (rolling-ball background subtraction → adaptive local thresholding → morphological cleanup → watershed) was tested directly (`src/segmentation/classical_watershed.py`) and works dramatically better than Cellpose-SAM zero-shot on this data:
- Periderm crop: 0.46% coverage / 79 instances (Cellpose) → **58.4% coverage / 20,464 instances** (classical watershed), visually clean cell-wall tracing through the previously-blank periderm band.
- Parenchyma crop (same code, no parameter changes): 48.1% coverage / 11,167 instances, visually accurate boundaries — confirms this isn't a periderm-only patch, it generalizes across tissue types with one parameter set.

**Phase 1 primary method is now this classical pipeline** (rolling ball → local adaptive threshold → morphological cleanup → distance-transform watershed), not Cellpose fine-tuning. Advantages: zero-shot (no annotated training data needed), deterministic, already validated on both tissue regimes present in `datasets/DO`. Remaining risks to check before treating this as final: whether `block_size`/`offset`/`rolling_ball_radius` need retuning per stain when moving to `datasets/examples/` (different stain families); watershed over-segmentation risk in regions with irregular internal cell texture (e.g. starch granules, noted as a specific failure mode requiring extra morphological cleanup in the source paper); Cellpose is kept as a fallback/comparison, not discarded.

**Diagnostic pass (stage-by-stage) — three real bugs found and fixed, none of them the starch-granule risk flagged above.** Built `src/segmentation/diagnose_stages.py`: runs `segment_classical(..., return_stages=True)` on a small, native-resolution patch (with margin-padding, see below) and renders every intermediate array (tissue mask, wall mask raw/cleaned, interior raw/filtered, distance transform raw/smoothed, watershed markers, final labels) as one panel grid, so a defect is attributable to a specific stage rather than only visible in the aggregate final count.

1. **`tissue_mask` broke on small tissue-only crops.** Otsu-on-grayscale (the original approach) forces a bimodal split of whatever brightness values are present in the given image — on a small crop containing zero real background, it still split the crop's own internal brightness range in half, wrongly marking real tissue corners as "background" (visually obvious as large black blobs in the `tissue mask` panel). Root-caused by direct measurement: real background samples measured saturation≈0.025/value≈0.96; this all-tissue crop measured mean saturation≈0.166 (even bright lumens carry a faint stain tint) with brightness ranging up to 1.0 — Otsu-on-grayscale is blind to the saturation channel, so it can't see that distinction, and picks a threshold relative to whatever's in the current image/crop rather than an absolute, transferable property. **Fixed** by reclassifying `tissue_mask()` on HSV saturation+brightness (`sat_thresh=0.08`, `val_thresh=0.85`: background = colorless AND bright) instead of a grayscale Otsu split — scale-invariant, unlike the old approach.
2. **Crop-boundary artifact, not a bug — but worth fixing in the diagnostic tool itself.** Even after fix #1, a tight crop still showed small residual unfilled patches: confirmed via direct comparison (same region sliced from a full-image-computed mask = 100% tissue, no holes) that these are large lumens clipped by the crop's own edge — their far side isn't visible, so they locally "touch the border" and `binary_fill_holes` correctly can't fill them, even though they're fully enclosed within the real full image. Fixed in `diagnose_stages.py` by processing a `margin_px`-padded region and cropping back down to the display window afterward, rather than segmenting the tight crop directly — the lesson generalizes: always segment on the full image (or a sufficiently padded region) and crop the *result*, never crop the raw image first.
3. **`peak_h` (absolute h-maxima threshold) silently dropped ~2/3 of real small cells — the actual dominant error, much larger than the over-segmentation risk this same parameter was originally tuned against.** Measured directly: at the old default `peak_h=4.0`, only 62 of 192 real-cell-sized (≥100px) interior regions in a test patch got a watershed marker at all — the other 130 weren't merged into neighbors, they were dropped entirely (zero label, invisible in the final result unless you count raw interior blobs vs. final instances, as done here: 292 raw interior components → only 67 final labels). A single **absolute** pixel-height threshold can't serve both small and large cells: low enough to recover small-cell markers, and internal texture/gradient bumps inside large cells clear the same absolute bar too, re-fragmenting real large cells (confirmed: a single 18,650px test vessel split into 5 pieces at an absolute threshold strong enough to recover small cells). **Fixed** by switching to a **per-component relative** h-maxima: normalize each connected interior blob's distance-transform values by *that blob's own max* before suppressing shallow peaks (`peak_h_frac=0.7`, replacing `peak_h`) — a small cell's own peak is always ~1.0 in its own normalized frame (survives regardless of absolute pixel size), while a large cell's internal texture must clear the same *fraction* of that specific cell's own radius to count as a separate peak. Verified on both test cases simultaneously: small-cell patch final instances 64→289 (matching the aggressive-but-over-splitting fix's recovery), while the 18,650px test vessel stays as exactly 1 label (vs. 5 at a naive low absolute threshold). This is a targeted fix, not a retry of the earlier-attempted-and-reverted "scale-adaptive local-cell-size estimation" (that used a smoothed neighbor-based size *estimate*, which misjudges atypically-sized cells; this uses each component's own exact measured max, no spatial estimation involved).

**Full-image impact (`DO_0000`):** 6,788 → 29,883 instances (median area 430px → 123px, consistent with recovering previously-missing small cells, not spurious noise — visually confirmed via `outputs/phase2/DO_0000_full_metric_area_classical.png`: every vessel now resolves as a single well-shaped unit, individual rows of vessels visible throughout the fan, and the periderm band shows crisp individual-cell striping instead of the previous coarse blur). This is a substantially better result than the pre-fix classical output, and now qualitatively comparable to or better than the `lightsheet_root` PlantSeg checkpoint from Phase 2.5 on the same image.

**Not yet re-validated after this fix:** the periderm-band coverage gap and the µm-calibration/starch-granule open questions from earlier in this phase were diagnosed against the *pre-fix* pipeline — revisit both against the current code before treating either as still-open exactly as originally described.

**Status: done.** `src/segmentation/classical_watershed.py` (`tissue_mask` HSV fix, `peak_h_frac` relative h-maxima fix, `return_stages` option), `src/segmentation/diagnose_stages.py` (new), diagnostic outputs in `outputs/phase1/DO_0000_classical_stages_diagnostic.png`.

**Update — batch-segmented the full `DO` set plus two additional datasets (`EH`, `VM`), zero-shot, no parameter retuning.** `datasets/` also contains `EH/` (82 images, ~2291x2206 to ~5227x5683) and `VM/` (67 images, up to ~10211x9889 — large enough to trip PIL's default decompression-bomb pixel-count guard, handled via `Image.MAX_IMAGE_PIXELS = None`), each with the same `inputimages/`+`annotations/`+train/test-split-`.txt` structure as `DO/`. Both stains are visually similar in overall hue balance to `DO`'s (blue-shifted RGB means), so `segment_classical`'s DO-tuned parameters (green-channel wall proxy, `tissue_mask` thresholds, `peak_h_frac=0.7`) were tried as-is rather than retuned — spot-checked first on one native-resolution 1200x1200 crop per dataset before committing to a full batch run (`outputs/phase1` — visual QC only, no saved crop-check file), then run over every image in each dataset.

New scripts: `src/segmentation/segment_full_image.py` (single DO image -> `outputs/phase1/<name>_full_labels.npy`, mirrors how `DO_0000_full_labels.npy` was originally produced), `src/segmentation/segment_all.py` (batch version, generalized to any `datasets/<NAME>/inputimages/` directory via a CLI arg, skips images that already have a saved label map, logs per-image timing/cell-count/failures to `outputs/phase1/segment_all_log.txt`).

Results, run zero-shot with 0 failures across all three datasets:

| Dataset | Images | Cell count range | Total batch time |
|---|---|---|---|
| `DO` | 64 | ~9,900–42,000/image | ~2h (62 newly run; `DO_0000`, `DO_0022` pre-existing) |
| `EH` | 82 | ~7,000–33,500/image | ~44 min |
| `VM` | 67 | ~24,000–85,000/image | ~2.5h |

All 213 `outputs/phase1/<name>_full_labels.npy` files together total 23GB on disk. No per-image failures logged, and no manual re-tuning was needed to get a plausible-looking result on either new dataset — but this is a zero-shot generalization result, not a validated one: no quantitative IoU/AP check was run against `EH`/`VM` (they have no per-cell instance ground truth, same limitation as `DO`), and no full-image visual QC pass (only the crop-level spot-check) was done before launching the batch. Treat per-image quality as unverified until spot-checked at full-image scale, same caveat as `DO`'s known periderm-band limitation (see above).

**Open question, not investigated:** `EH`/`VM` species/tissue identity is unconfirmed. Given the Phase 2.5 correction that `DO` is actually a *Dryas octopetala* dendrochronology cross-section (INBD paper) rather than potato tuber as originally assumed, and that `EH`/`VM` follow the identical species-code-prefixed dataset structure, they are plausibly other species from the same INBD tree-ring dataset rather than an unrelated source — not confirmed against the paper's actual sample list, so treat as an open question rather than assumed fact (same lesson as the earlier DO mislabeling).

---

## Phase 2 — Geometry features & structural graph
**Goal:** per-cell feature table + adjacency graph, deterministic and inspectable.

- From each instance polygon compute: area, perimeter, equivalent diameter, major/minor axis, elongation ratio, orientation angle, solidity, circularity, eccentricity, centroid.
- Fit image-level radial coordinate system (centroid of the section) → add radial distance + angular position per cell.
- Build Region Adjacency Graph (RAG) from touching masks; edge features: shared-wall length, centroid distance, relative orientation delta, radial-vs-tangential flag.
- Export as `torch_geometric.data.Data` objects (one per image or per tile-subgraph).
- Build a quick visualization utility: color cells by elongation/orientation/radial-distance to sanity-check against the visible tissue rings.
- **Deliverables:** `src/graph/features.py`, `src/graph/build_graph.py`, `src/graph/viz.py`.
- **Exit criteria:** overlay plots show expected patterns (e.g. small elongated tangentially-oriented cells near vascular ring, large isodiametric cells in pith) — this is your correctness check before any model training.

**Status: done for `DO_0000` (123,782 cells post-cleanup), two bugs found and fixed along the way:**
1. **Background contamination.** The classical watershed pipeline (Phase 1) segmented the white slide background outside the tissue into thousands of meaningless "cells" with degenerate feature values (elongation up to ~1e8). Fixed by adding an Otsu-based tissue mask (`tissue_mask()` in `src/segmentation/classical_watershed.py`) that restricts segmentation to the actual tissue region. A residual handful of degenerate slivers along internal tissue tears/voids (real damage in the specimen, correctly inside the tissue mask) were then dropped via a minimum area/minor-axis filter in `extract_cell_features()`.
2. **Adjacency undercounted.** Initial region-adjacency used direct pixel touching, giving an average node degree of ~1.4 — far below the ~5-6 expected for a planar cell tessellation — because cell walls are several pixels thick, so interiors on either side of a wall never touch directly in the label image. Fixed by growing each label into its surrounding wall band (`skimage.segmentation.expand_labels`, `wall_expand_px=4`, tuned from a measured ~4px average wall thickness) before checking adjacency. This brought average degree to ~5.15, matching expectation. This was a correctness-critical fix given connectivity/adjacency was the original motivating requirement for the graph representation.

QC overlays (`outputs/phase2/DO_0000_qc_overlay_v3.png`) now show the expected radial gradient, a coherent radial-fan orientation pattern, and a sane elongation distribution.

**Update — watershed over-segmentation diagnosed and largely fixed for parenchyma, periderm regressed.** Visual inspection (not just aggregate stats) revealed the real problem behind the implausible µm sizes above: severe watershed over-segmentation, not primarily starch granules. A tight native-res parenchyma crop with ~15-20 real cells by eye was producing 504 watershed regions. Root cause: (1) internal wall-color gradients (multiple stain layers within one wall) got picked up by the adaptive threshold as multiple separate wall lines, and (2) noisy cell-interior texture created multiple close local maxima in the distance transform, seeding multiple markers per real cell.

Fixes applied in `src/segmentation/classical_watershed.py`:
- Gaussian-smoothed the distance transform before peak detection, and added `h_maxima` suppression of shallow/spurious peaks combined with a larger `peak_min_distance`, before running watershed — reduced the same parenchyma crop from 504 → 15 regions, now matching real cell boundaries one-to-one.
- Replaced the fixed-offset local-mean threshold (`threshold_local`, hand-tuned `offset=-2`) with local Otsu (`skimage.filters.rank.otsu`, disk(15) footprint) — self-calibrating per neighborhood, visually more consistent across both tissue types than global Otsu (illumination-sensitive) or Li's method (broken walls), tested via a direct 2x2 comparison (`outputs/phase2/threshold_method_comparison.png`).

**Regression:** the same fixed parameters (disk(15) local-Otsu footprint, `min_cell_px=15`, peak-suppression constants) that fixed parenchyma under-serve periderm — large regions of clearly-cellular periderm tissue now go completely undetected (too fine-scale for these constants), a relocated version of the original multi-scale tension rather than a new bug.

**Attempted: scale-adaptive local-cell-size estimation (reverted).** Tried making the wall threshold and watershed marker step spatially adaptive — a per-pixel local-scale map (smoothed local max of the distance transform, not the naive area-weighted mean which is dominated by thin near-wall pixels and badly underestimates true cell radius) feeding a multi-footprint local-Otsu blend and a greedy spatial non-max-suppression for peaks. Result: worse, not better (338/511 regions on the two test crops, vs. the original 504/654 and the fixed-parameter version's 15/14). Root cause understood, not just a tuning miss: pure spatial distance between peaks isn't equivalent to the *prominence* (saddle-depth) suppression `h_maxima` was doing — a real cell's interior often has multiple local maxima farther apart than any reasonable suppression radius, so spatial NMS alone can't tell "two real cells" from "one irregular cell with two bumps." A correct fix needs region-adjacency-graph merging (over-segment via watershed, then merge adjacent regions by local saddle-strength relative to each pair's own scale) — meaningfully more implementation than a peak-detection tweak, so deferred rather than pursued further now.

**Decision: reverted to `segment_classical`** (local Otsu + h-maxima peak suppression) as the working Phase 1/2 method — solid on parenchyma, periderm coverage remains a known, documented limitation. RAG-based merging is a candidate for a later pass if periderm coverage becomes a blocker; not pursued now in favor of moving to Phase 3. Absolute µm size figures remain provisional pending this.

**Correction — full-image "coverage collapse" was a measurement bug, not a real regression.** After the fix above, re-running `segment_classical` on the full `DO_0000` image dropped the cell count from a stale 130,828 (pre-fix, over-segmented) to 6,752 — expected, since the fix removes spurious fragments. But a `tissue_mask`-based coverage check appeared to show 72-80% of tissue going completely unlabeled at full-image scale, triggering an extended (5+ diagnostic) investigation into `rolling_ball`/thresholding/marker-placement/watershed scale behavior. All were individually ruled out as matching small-crop behavior almost exactly — correctly, as it turned out, because **the diagnostic itself was flawed**: `tissue_mask()` marks all non-background tissue (interiors *and* walls) as foreground, while `labels>0` only covers interiors by design (walls are meant to stay unlabeled). Re-running the same "unlabeled" check on the *already-validated* standalone crop (the one visually confirmed to have correct, complete cell boundaries) showed the same ~48% "unlabeled" figure — proving this was never a full-image-specific problem, just a wrong metric. A tile-based segmentation workaround was built and then removed once this was discovered, since it was solving a non-existent problem.

**What was real:** the effective gap between labeled cell interiors grew from ~4px to ~16px after the over-segmentation fix (not wall thickness alone, but wall thickness plus the normal unlabeled rim near walls that watershed leaves on both old and new segmentation equally). `wall_expand_px=4` in `build_graph.py` was stale from the old segmentation and needed to be ~22 to recover avg degree ~5 on the new one — **`wall_expand_px` must be re-measured whenever the segmentation pipeline changes materially, not treated as a fixed constant.** Fixed in `src/graph/build_graph.py` (default now 22, with a docstring warning against treating it as fixed).

**Current state, confirmed visually.** Full-image overlay of the corrected segmentation (`outputs/phase2/DO_0000_full_overlay_corrected.png`, 6,752 cells) shows complete, plausible boundary coverage across essentially the whole section — including some boundary detection now in the periderm band, not the total blank seen pre-fix — with no large unlabeled gaps by eye. This is the segmentation + graph state Phase 3 was retrained against (see below). Treat `outputs/phase1/DO_0000_full_labels.npy` as the current source of truth for `DO_0000`'s segmentation; anything computed from an earlier copy of that file (any output predating this correction) is stale.

**Open issue: physical (µm) size calibration is unresolved.** `datasets/DO` ships no scale metadata (no EXIF, TIFF resolution tags are uncalibrated `(1,1)`). A `DO_UM_PER_PIXEL = 0.1629` constant was derived from the source paper's stated 3x-zoom/890×740µm² field of view, but applying it to current segmentation output gives a median cell diameter of ~1.1 µm — implausible for potato parenchyma (well documented at ~100-170 µm, among the largest known plant cells). The raw-pixel median equivalent diameter is only ~8px regardless of scale choice, a ~2-orders-of-magnitude gap too large to be calibration error alone — the likely cause is **watershed over-segmentation from starch granules** (internal cell contents creating false internal wall edges), which the source paper explicitly flags as requiring dedicated morphological cleanup that `classical_watershed.py` doesn't yet do. `um_per_pixel` conversion is implemented and wired through (`src/graph/features.py`, `src/graph/build_graph.py`), but absolute size figures should be treated as provisional until this is resolved — needs either starch-granule-aware segmentation cleanup or an independent in-image scale reference to separate the two open questions (is it the scale constant, the over-segmentation, or both).

---

## Phase 2.5 — PlantSeg baseline & boundary-detection validation
**Goal:** evaluate an off-the-shelf pre-trained boundary-detection model (PlantSeg, Wolny et al., eLife 2020) on shrub cross-section imagery as both a baseline for comparison and as a test harness for the boundary-detection + graph-partitioning pipeline you'll need in Phase 4.

**Context:** PlantSeg is a 3D volumetric cell segmentation tool trained on confocal/light-sheet plant tissue imagery (Arabidopsis ovules, lateral root primordia). While this project's data is 2D cross-sections, PlantSeg's two-step pipeline — (1) 3D U-Net boundary detection, (2) graph partitioning to convert boundaries into instance labels — maps directly onto Phase 4's round-trip validation (field → recovered graph). Testing it here:
  - Provides a quantitative/qualitative baseline against `classical_watershed.py`
  - Validates that the graph-partitioning downstream machinery works before Phase 3/4 modeling
  - Tells you whether a generic plant tissue segmenter generalizes to your specific shrub cross-sections or if specialized tree-ring methods (e.g., INBD) are required

**Deliverables:**
- `src/segmentation/plantseg_eval.py` — loads pretrained `pytorch3dunet` boundary-detection checkpoints (PlantSeg's model zoo, downloaded directly rather than installing the full app) and runs tiled inference + watershed instance extraction on test crops
- Side-by-side visual comparison: PlantSeg checkpoints vs. `classical_watershed.py` on the same test crops
- Decision flag per checkpoint: does its output look plausible on your data?

**Steps:**
1. ~~Install `plantseg` package~~ — **superseded**: the full PlantSeg app requires conda-forge-only GUI/heavy dependencies (`napari`+Qt, `vigra`, `bioimageio.core`) not available on PyPI, a multi-GB install pulling in a full GUI stack just for headless inference. Used `pytorch3dunet` instead (Wolny's own underlying model-code package, pure PyTorch + numpy/scipy/scikit-image, pip-installable, no GUI deps) — this is the actual boundary-detection network PlantSeg wraps, not a reimplementation.
2. Downloaded two pretrained 2D U-Net boundary-detection checkpoints directly from PlantSeg's model zoo (Zenodo-hosted, ~21.8MB each, no auth): `confocal_2D_unet_ovules_ds2x` and `unet2d-lateral-root-lightsheet` (`models/pytorch3dunet/`). Architecture confirmed by loading state_dicts with `strict=True` (not assumed from doc examples, which referenced a different `f_maps` value than what the checkpoints actually contain): `UNet2D(in_channels=1, out_channels=1, f_maps=64, layer_order="gcr", num_groups=8, num_levels=4)`.
3. Ran tiled inference (256px patches, 192px stride, overlap-averaged) on the same crops used by `classical_watershed.py` (periderm + an interior/parenchyma crop) from `datasets/DO` test split.
4. Converted boundary-probability maps to instance labels via the same distance-transform-watershed post-processing as `classical_watershed.py`, so the comparison isolates the boundary *detector*, not the instance-splitting step.
5. Compared visually against classical watershed on the same crops (`outputs/phase2/DO_0022_{periderm,interior}_plantseg_vs_classical.png`).

**Two bugs found and fixed during evaluation:**
1. **Polarity mismatch.** Both checkpoints are trained on fluorescence data (bright walls on dark background); this project's stained brightfield images are the opposite polarity (dark walls, light background). `confocal_ovules` on raw (uninverted) grayscale collapsed to a single instance (boundary-probability max ~0.085, far too low to threshold). Confirmed by direct test: inverting intensity (`1 - gray`) raised the max to ~0.43 and produced a real wall-tracking signal. `lightsheet_root` happened to already produce a usable signal without inversion — not evidence it generalizes better, more likely incidentally less polarity-sensitive. Fixed via an explicit per-checkpoint `INVERT_INPUT` flag in `plantseg_eval.py` (`confocal_ovules: True`, `lightsheet_root: False`), determined empirically, not assumed.
2. **Fixed 0.5 threshold silently zeroed an out-of-domain model's output.** A flat cutoff (reasonable for a well-calibrated in-domain model) is wrong when a model's output scale on unfamiliar data is uncalibrated (~0.085-0.43 max, not ~1.0). Replaced with a per-image Otsu threshold on the probability map.

**Result:** the two checkpoints diverge sharply on this data.
- **`lightsheet_root` generalizes strikingly well** — its boundary-probability map cleanly and continuously traces real cell walls across both crops, including in the periderm band (a documented weak/under-detected spot for `classical_watershed.py`, see Phase 1). Visually plausible, arguably cleaner than classical watershed's more fragmented wall detection in that region.
- **`confocal_ovules` remains weak** even after fixing both bugs above — Otsu thresholding keeps it from collapsing to nothing, but the resulting instance boundaries follow noise texture rather than real cell walls (blocky, jagged regions not aligned to visible walls). Not a viable boundary detector on this data as-is.

**Exit criteria: met.** Inference runs successfully end-to-end on MPS, output is directly compared against `classical_watershed.py` on the same images, and there's a documented, checkpoint-specific answer to "does a pretrained plant-tissue U-Net generalize to our data?" — yes for `lightsheet_root`, no for `confocal_ovules`, with the reason understood (probably incidental polarity/contrast robustness of the light-sheet checkpoint, not a deeper domain-match story, since light-sheet and confocal are both fluorescence and equally far from brightfield stains in modality).

**Open follow-up (not pursued in this pass):** `lightsheet_root`'s strong result is a candidate replacement or ensemble input for `classical_watershed.py`'s boundary-detection step specifically in the periderm band; the instance-splitting (watershed) step is currently shared/unchanged between both methods, so a fairer full pipeline comparison would also swap in PlantSeg's actual graph-partitioning step (GASP/Multicut) rather than reusing the classical watershed post-processing as done here for a controlled comparison.

**Correction (mid-Phase 2.5): this is not potato tuber tissue.** User identified the true source: a shrub/woody branch cross-section used for dendrochronology (tree-ring dating) — see `literature/2212.03022v2.pdf` (Gillert et al., "Iterative Next Boundary Detection," INBD), whose `DO` dataset subset (*Dryas octopetala*) matches the `datasets/DO/` naming in this repo. Confirmed visually once full-image overlays were produced: concentric growth-ring structure, a central pith, and radiating rays are unmistakable in `DO_0022`'s full-image boundary overlay — not consistent with potato parenchyma. This does not invalidate the PlantSeg checkpoint comparison itself (that was testing generic cross-domain generalization, not potato-specific behavior), but the `DO_UM_PER_PIXEL` calibration constant and all potato-anatomy framing earlier in this document (Phase 0/1 "Reference data," periderm/cortex/pith terminology borrowed from the Biswas & Barma potato paper) should be treated as provisionally mislabeled pending a proper re-read of `datasets/DO/` against the INBD paper's actual sample descriptions. Not corrected retroactively above to preserve the historical record of what was believed at each point; treat the potato-specific claims in Phases 0–2 as superseded by this note.

**Extended Phase 2.5 — Full-image testing and diagnostic work:**
- Full-image overlay, instance maps, and lumen-marking visualizations at native resolution (5578×5290).
- Surfaced bugs invisible at crop scale: (1) missing tissue mask on full path — reused `tissue_mask()` from Phase 1; (2) over-segmentation from plain `peak_local_max` — fixed by adding `h_maxima` peak suppression (same as Phase 1).
- **New `tissue_mask()` bug found via diagnostic tool** (`src/segmentation/diagnose_stages.py`): Otsu on grayscale splits any image in half, not discriminating tissue vs. background when both are present in a crop. On tissue-only patches, it marks real tissue as "background" (confirmed: 96% tissue without padding, 100% with margin). Fixed by switching to HSV saturation+value (tissue is colored+dark, background is colorless+bright) — scale-invariant and works on any crop size. Shared fix benefits both `classical_watershed.py` and PlantSeg paths.
- **Vessel-exclusion bug:** `tissue_mask()` used `remove_small_holes()` with fixed `area_threshold=5000`, excluding large vessels (same color as background, can't be filled if they exceed the threshold). Switched to `binary_fill_holes()` which fills all enclosed holes regardless of size — fixes vessels without breaking background exclusion.
- Instance counts after fixes: confocal 16,121→8,679; lightsheet 33,756→12,571 (consistent reduction across both, confirming fixes targeted real problems, not dataset-specific artifacts).

**Classical Watershed vs. PlantSeg full-image comparison (DO_0024, 2361×2333 central region):**
- Classical watershed: 9,841 instances. Clean wall detection, individual cells distinguishable, moderate over-segmentation (internal texture creates some fragments).
- PlantSeg `lightsheet_root`: 3,413 instances. Significant under-segmentation — many real cells merged into "super-cells," indicates poor generalization from fluorescence training data to brightfield stain.
- **Verdict:** Classical watershed outperforms PlantSeg on this brightfield wood-cross-section data. PlantSeg is useful as a diagnostic comparison but not a replacement for the domain-tuned watershed approach.

**Status: done.** `src/segmentation/plantseg_eval.py`, `src/segmentation/diagnose_stages.py`, `models/pytorch3dunet/*.pytorch` (2 checkpoints), `src/segmentation/classical_watershed.py` (shared `tissue_mask` fix), `src/graph/features.py` (`max_area_log_std` outlier filtering), outputs in `outputs/phase2/` and `outputs/phase1/`.

---

## Phase 3 — Graph-VAE representation learning
**Goal:** latent embedding of tissue composition, validated against known tissue zonation.

- Subgraph sampling strategy (k-hop neighborhoods or random-walk subgraphs, ~200–500 samples/image) to get enough training examples from few whole images.
- Encoder: 2–3 layer GAT/GraphSAGE → attention pooling → VAE bottleneck (mean/logvar).
- Decoder: MLP/GNN decoder reconstructing node features; inner-product decoder for edge/adjacency reconstruction.
- Loss: node feature reconstruction + edge (adjacency) reconstruction + KL term; optionally an auxiliary tissue-type classification head if you hand-label a few regions per image (epidermis/cortex/vascular/pith) to anchor the latent space semantically.
- Train on the 3–5 available images (heavy rotation/reflection augmentation — valid here given radial symmetry).
- Evaluate: cluster the latent space (UMAP/PCA), check clusters correspond to visible tissue rings; check reconstruction quality on held-out subgraphs.
- **Deliverables:** `src/models/gvae/`, trained checkpoint, latent-space visualization notebook.
- **Exit criteria:** latent clusters visibly separate by tissue type/region without being told to; reconstructed node features/adjacency close to input.

**Status: done for a first pass on `DO_0000`.** Built `src/models/gvae/model.py` (2-layer GAT encoder, attention pooling, VAE bottleneck, GNN decoder reconstructing node features + inner-product edge decoder), `sampling.py` (k-hop subgraph sampler, num_hops=3, up to 300 nodes/subgraph), `train.py`, `eval.py`. Trained 800 steps on MPS (confirmed device placement directly — node/edge reconstruction loss both dropped over training, KL grew as expected for a VAE). Latent space evaluated via SVD-based PCA (no sklearn dependency) over 800 held-out sampled subgraphs, colored by subgraph-mean feature values:
- `radial_distance`: clean, smooth gradient across the latent projection — corr(latent PC1, mean radial_distance) = **-0.83**. The model learned radial tissue position from local graph structure alone, unsupervised — this is the core Phase 3 exit criterion, met.
- `area`: partial secondary gradient (expected — cell size and radial position are naturally correlated in this tissue).
- `elongation`: no clear structure yet — plausibly needs more training steps; not investigated further in this first pass.

Not yet done: multi-image training (currently single-image, `DO_0000` only), the optional auxiliary tissue-type classification head, and reconstruction-quality metrics beyond the training loss curve.

**Retrained after the Phase 2 `wall_expand_px` fix** (see Phase 2 correction above — graph went from the stale 130,828-node over-segmented version to the corrected 6,752-node one, avg degree ~5.05). Latent-space correlation with radial position weakened but remains real: corr(latent PC1, mean radial_distance) = **-0.44** (down from -0.83 on the stale graph), still visible as a gradient in the PCA plot. Plausibly weaker partly because the corrected graph has less diverse local neighborhood structure to sample from at this smaller scale (6,752 vs. 130,828 nodes, same single image) — not yet investigated further; multi-image training would be the natural next test of whether this strengthens with more data.

**Multi-image batch training implemented and run across all 213 cached graphs (`DO`+`EH`+`VM`, see Phase 1 update).** New: `src/graph/build_all_graphs.py` (caches each image's `(Data, node_df)` to `outputs/phase2/graphs/<name>.pt`, ~2-30s/image depending on cell count, skips already-cached images) and `src/models/gvae/train.py::train_multi()` / `eval.py::evaluate_multi()` (feature standardization pooled across *all* images, not per-image, so subgraphs from different specimens land in a shared feature space; original single-image `train()`/`evaluate()` kept unchanged for backward compatibility; multi-run artifacts get a `_multi` filename suffix so they don't clobber the single-image checkpoint).

Ran 1500 steps over 4,815,125 total nodes / 27,855,706 total edges (loss curve healthy: node loss trending down with noise, KL stabilized ~0.65-0.8). Result:
- corr(latent PC1, mean radial_distance) = **-0.845** — stronger than any single-image run so far (vs. -0.44 DO_0000-corrected, 0.685 DO_0022 alone). The radial-position signal is real and gets *more* legible, not less, with more images and more diverse local structure to sample from.
- **New diagnostic added and a real caveat found:** fraction of latent PC1 variance explained by *which source image* a subgraph came from (vs. within-image composition) = **0.387** — not dominant, but not negligible either. Some of what the latent space is separating on is "which of the 213 specimens is this," not purely tissue composition, which is exactly the failure mode a composition-focused latent space shouldn't have. Plausible causes not yet isolated: real cross-dataset differences (DO/EH/VM likely different stains/species, still unconfirmed per the Phase 1 note), or an artifact of pooling absolute-scale features (raw pixel `area`, `equivalent_diameter`, etc.) across images at very different resolutions/FOVs without per-image or um-calibrated normalization.
- `elongation`/`area` PCA panels are visually less informative than the single-image runs — most points compress near the low end of the color scale with a few extreme outliers dominating the colorbar range, consistent with the same absolute-scale-pooling explanation above (an outlier subgraph from a large-cell region on one image can dwarf the rest of the range across 213 images of varying pixel resolution).
- Checkpoints: `outputs/phase3/gvae_multi.pt`, `feature_scaler_multi.npz`, `latent_space_pca_multi.png`.

**Control run — `DO`-only, isolates cross-dataset effect from scale-pooling artifact.** Ran the same `train_multi`/`evaluate_multi` restricted to just `DO`'s 64 cached graphs (`train.py`/`eval.py` now take an optional dataset-prefix arg, e.g. `DO`, filtering the glob and tagging checkpoints accordingly — `outputs/phase3/gvae_DO.pt` etc.). Result: corr(latent PC1, radial_distance) = 0.623, and **between-image variance fraction drops to 0.138** (vs. 0.387 on the full DO+EH+VM mix). Since `DO` alone still has real pixel-resolution variation across its 64 images (per `do_dataset.py`, ~2964x2723 to ~8786x9815) and still shows a nonzero 0.138, scale-pooling is a real but secondary contributor -- the much larger jump to 0.387 when `EH`/`VM` are added is better explained by a genuine cross-dataset effect (different stain/species/imaging setup) than by scale pooling alone. This doesn't yet distinguish "real biological/imaging difference the model should be allowed to see" from "an artifact the model shouldn't be keying on" -- both are consistent with this result -- but it does rule out scale-pooling as the primary cause.

**`EH`/`VM` species identity confirmed against the source paper (literature/2212.03022v2.pdf, Ganz et al., INBD, Table 1).** `DO` = *Dryas octopetala*, `EH` = *Empetrum hermaphroditum*, `VM` = *Vaccinium myrtillus* — three distinct dwarf-shrub species, not stain/imaging variants of one species. Confirmed unambiguously: each species' train+test image count in Table 1 matches this repo's count exactly (DO 22+42=64, EH 24+58=82, VM 22+45=67). The paper itself independently notes "EH and VM show some level of similarity to each other... DO on the other hand is visually dissimilar" — a real prior about expected cross-species structure, not just a naming lookup.

**Correction — species identity does NOT explain the between-image variance; the earlier DO-only-control interpretation was too simple.** Re-ran `evaluate_multi` on the existing `gvae_multi` checkpoint (no retraining) with a new species-colored PCA panel and a between-*species*-variance decomposition (3 groups) alongside the existing between-*image* one (213 groups). Result: **only 6.7% of PC1 variance sits between the 3 species**, vs. 44.2% between individual source images — the species-colored panel shows heavy visual overlap between DO/EH/VM, not the two-cluster separation (DO vs. EH+VM) the paper's own "DO is visually dissimilar" note would predict if species identity were the dominant factor. So the earlier reasoning from the DO-only control (13.8% within-DO vs. 38.7% full-mix, taken as evidence of a real cross-dataset effect) was directionally right that *something* real changes when EH/VM are added, but wrong to attribute it mainly to species-level separation — the dominant factor is per-specimen idiosyncrasy (individual growth/imaging variation), which just has more room to accumulate across 213 images than 64. Also a plainer statistical confound worth naming: a between-group variance fraction computed over 213 small groups is not directly comparable to the same statistic over 64 groups even under a null of no real per-group effect, since finite-sample noise in each group's mean inflates the "between" term more as group count grows and per-group sample size shrinks.

**t-SNE added alongside PCA to check this isn't a linear-projection blind spot.** New dependency `scikit-learn` (only used for `sklearn.manifold.TSNE`; PCA stays the existing no-dependency SVD implementation since its axes need to stay linearly interpretable for the corr()/variance-fraction checks — t-SNE is nonlinear, so it's used for visual cluster inspection only, plotted as a second `latent_space_tsne_{tag}.png` with the same 5-panel layout). Result: t-SNE agrees with PCA, not just reproduces the same linear-projection limitation — the species panel shows the same DO/EH/VM interleaving (t-SNE's usual local-island fragmentation, but no clean 3-way separation), while the radial_distance panel still shows a coherent gradient across the embedding. Re-running `evaluate_multi` (fresh random subgraph sample) also gave a consistent between-species figure (2.7%) alongside a consistent between-image figure (35.4%) — same conclusion as before, not a one-off.

**Root cause found: per-image imaging/segmentation variation, specifically raw-pixel resolution and cell count -- not species, not stain color.** New `src/models/gvae/diagnose_image_variance.py`: for each of the 213 images, samples 20 subgraphs to get a stable per-image mean latent vector, then correlates each image's latent PC1 position against ten cheap per-image covariates (image resolution/megapixels, mean RGB/HSV from a downsized thumbnail, `num_cells`, `mean_area_px`, `mean_equiv_diam_px`, `cell_density`). Result (`outputs/phase3/image_variance_diagnostic_multi.{csv,png}`):
- **`num_cells` corr = +0.781, `megapixels` corr = +0.723** -- by far the strongest predictors, far ahead of any color/stain covariate (`mean_r/g/b`, `mean_saturation`, `mean_value` all |r|<0.25).
- **Joint R^2 of all 10 covariates predicting per-image latent PC1 = 0.864** -- imaging/segmentation-derived stats alone explain the great majority of the between-image latent variance found earlier (35-44%), leaving little room for a genuine biological/species effect (consistent with species only explaining 2.7-6.7%, itself plausibly just a downstream correlate of species differing somewhat in typical image resolution, not a direct species effect).

**Mechanism, not just correlation:** confirmed there's no physical (um) scale calibration available for *any* of the three datasets -- the source paper (`literature/2212.03022v2.pdf`) reports no pixel-to-mm conversion, and DO's own provisional `DO_UM_PER_PIXEL` constant (Phase 1/2) is unresolved and unapplied by default. `build_graph.py`'s node features (`area`, `equivalent_diameter`, `major_axis_length`, etc.) are therefore raw pixel values, and `train_multi`'s global standardization pools these directly across 213 images spanning ~2.5x to 10x native resolution differences (do_dataset.py's noted ~2964x2723 to ~8786x9815 range applies dataset-wide, not just to DO). A cell's absolute pixel area is mostly telling the model which image's native resolution it's looking at, not how big the cell really is -- exactly the kind of scale artifact flagged as a risk back when multi-image training was first implemented, now precisely localized rather than just suspected.

**Confirmed exhaustively: no physical scale calibration is published anywhere for `DO`/`EH`/`VM`.** Checked the dataset's actual source (`github.com/alexander-g/INBD`, the "MiSCS" -- Microscopic Shrub Cross Sections -- dataset by Anadon-Rosell et al.): repo README, the arXiv paper (already in `literature/`), and the CVPR 2023 supplementary PDF (not included in the arXiv version, fetched separately from `openaccess.thecvf.com` -- also checked for microscope/magnification/pixel-size/scale-bar mentions, found none). Table 1's "Average diameter" column (DO 3700px, EH 3260px, VM 3979px) is pixel-only, not a calibrated measurement. Confirmed the locally-downloaded `datasets/DO|EH|VM/` directories already match the released zip contents exactly (same `inputimages/`/`annotations/`/split-`.txt` structure, no extra metadata file). The only path to real um/pixel values is contacting the data collector directly (README lists `a.anadon at creaf.uab.cat`) -- out of scope to pursue automatically. **Given that, the fix for the resolution-leak finding above should be per-image z-scoring or scale-invariant-only features (option a/b noted above), not physical-unit calibration -- that path is confirmed closed, not just unresolved.**

**Not yet done:** the fix -- either (a) per-image z-scoring of size-related features before pooling (removes absolute-resolution differences, keeps within-image relative structure) or (b) drop raw-pixel size features from the node feature set and keep only scale-invariant ratios (`elongation`, `circularity`, `solidity`, `eccentricity`, normalized radial position) -- has not been implemented or re-evaluated; the optional auxiliary tissue-type classification head; reconstruction-quality metrics beyond the training loss curve.

---

## Phase 4 — Field representation & reconstruction round-trip
**Goal:** bridge from graph representation to a rasterized field DDIM can operate on, and back.

- Rasterize each graph into a multi-channel field: signed-distance-to-wall, local orientation, local size, (optional) tissue-type/stain channel — computed from node/edge attributes over the image grid.
- Build the inverse: field → seed points (local SDF maxima) → watershed/Voronoi tessellation → recovered polygons → recovered graph.
- Validate round-trip: real graph → field → reconstructed graph; compare node feature distributions (area, elongation, orientation histograms) and graph statistics (degree distribution) between original and round-tripped graph.
- **Do not proceed to Phase 5 until round-trip error is low** — this isolates field-representation bugs from diffusion-model bugs.
- **Deliverables:** `src/graph/field.py`, `src/graph/tessellate.py`, round-trip validation report.
- **Exit criteria:** round-tripped graph statistics close to original (define explicit thresholds, e.g. <10% distributional shift on area/elongation histograms).

---

## Phase 5 — Conditional DDIM on fields
**Goal:** proof-of-concept generative model, conditioned on GVAE composition latent.

- Crop fields into patches (128×128 or 256×256) for training; this is where most of your effective training data comes from at this stage.
- UNet backbone over field channels, FiLM or cross-attention conditioning on the Phase 3 latent code.
- Use `diffusers.DDIMScheduler` (or custom) for the noise schedule/sampler; start with a small model, short training run to prove the loop works end-to-end before scaling.
- Sample → tessellate (Phase 4 inverse) → visualize generated cell layout.
- Evaluate generated vs. real: node-feature distribution comparison, graph statistic comparison (same metrics as Phase 4 round-trip check).
- **Deliverables:** `src/models/ddim/`, sampling script, generated-vs-real comparison report.
- **Exit criteria (prototype-level):** generated patches are locally plausible (cell-like tessellation, reasonable size/elongation stats) — full novel whole-structure generation is explicitly out of scope until more data is collected (see Phase 6).

---

## Phase 6 — Data scale-up & generalization
**Goal:** move from "pipeline proven" to "results that generalize."

- Identify target dataset size: ~20–50 sections for a GVAE latent space that generalizes across biological variability; ~30–100+ diverse specimens before DDIM output is meaningfully non-memorized (per earlier data-sizing discussion).
- **Update:** `datasets/DO` (64), `EH` (82), and `VM` (67) — 213 images total — are now all batch-segmented (see Phase 1 update above), well past the ~20-50 section lower bound and into range of the ~30-100+ upper end cited for non-memorized DDIM output. This changes Phase 6 from "source more data" to "validate what's already segmented and adapt Phases 2–5 to run across it" — Phase 3 (Section 3 status) is still single-image-trained; extending its subgraph sampler to draw from all 213 images is the next concrete step, not further data sourcing.
- Caveat carried over from the Phase 1 update: `EH`/`VM` segmentation is zero-shot and not yet quality-validated the way `DO` was (no IoU/AP check possible, no full-image visual QC pass done) — worth a spot-check pass across a sample of each before treating all 213 as trustworthy training data, not just the crop-level check already done.
- `EH`/`VM` species/tissue identity is unconfirmed (see Phase 1 update) — if they turn out to be different species entirely, that's useful cross-species diversity for this phase's goal; if they're the same species as `DO` differently stained, they don't add real diversity even though they add image count.
- Re-run Phases 1–5 at scale; re-validate exit criteria at each phase with the larger dataset.

---

## Phase 6.5 — Learned segmenter via pseudo-labeling (deferred)
**Goal:** replace the classical watershed pipeline's per-image/per-stain parameter tuning with a learned model that generalizes to new cell types/stains without retuning, once that becomes the actual bottleneck (not before).

- Rationale (from Phase 1 discussion): the classical rolling-ball + adaptive-threshold + watershed pipeline works well zero-shot on `datasets/DO`, but its `rolling_ball_radius`/threshold parameters may need per-stain retuning as more species/stains are added, and per-pixel watershed is slower than a forward pass at full-section resolution. Deferred rather than done now because it adds setup cost with no accuracy benefit until one of those becomes a real constraint.
- Approach: run the classical pipeline across the growing image set to generate instance masks as **pseudo-labels** (no manual annotation needed), then fine-tune Cellpose (or a lighter/faster network) on those pseudo-labels.
- This directly targets generalization to *other cell types* — the pseudo-label source data should span whatever tissue types/stains have been collected by that point (periderm, parenchyma, stem-section vascular tissue, etc.), so the learned model inherits the classical method's cross-tissue coverage rather than Cellpose's original fluorescence-cell-dominated training distribution (see prior discussion on Cellpose's original training data).
- Validate the learned model against the classical pipeline's own outputs (agreement) and against any hand-corrected tiles (accuracy), and against inference speed on full-resolution sections (the actual motivating metric).
- **Trigger to start this phase:** either classical-pipeline parameter tuning is consuming real time across new stains, or full-resolution watershed speed becomes a bottleneck for the dataset size in play — not on a fixed schedule.

---

## Phase 6.6 — U-Net cell-boundary segmenter (paper-matched DNN alternative to watershed)
**Goal:** add the source dataset paper's own DNN segmentation method (Biswas & Barma, *Scientific Data* 2020, `literature/s41597-020-00706-9.pdf`) as a selectable alternative to `classical_watershed.py`, not a replacement — same role as Phase 6.5's learned segmenter, but matching the paper's specific architecture/training recipe instead of fine-tuned Cellpose.

- **Architecture (from the paper, Methods → "Cell segmentation"):** U-Net (Ronneberger et al.), binary cell-boundary output. Trained on 512×512 patches generated by subdividing each full-resolution image + ground-truth pair into 20 sub-images. Two parallel input variants tested (raw RGB vs. normalized) — normalized input gave the better result (mean IoU 0.7020 vs. 0.6964 raw, Table 5). Optimizer: Adam, learning rate 10⁻¹ (as stated in the paper — unusually high, verify empirically rather than assuming a typo), with early stopping on a validation split.
- **Ground-truth data gap (confirmed, not assumed):** `datasets/DO/annotations/` only ships 9-class semantic tissue-region labels (periderm/cortex/vascular/pith etc., verified via direct pixel-value inspection — values `-1..8`), **not** the per-cell binary boundary masks the paper's U-Net was actually trained on. The paper's real boundary ground truth (≈60 hand-corrected images, see Fig. 6's rolling-ball → adaptive-threshold → morphological-cleanup → manual-correction pipeline) only exists bundled inside the full figshare deposit as a single **37.5 GB `Microscopy_image_dataset.rar`** (article [12206270](https://springernature.figshare.com/articles/dataset/A_large-scale_optical_microscopy_image_dataset_of_potato_tuber_for_deep_learning_based_plant_cell_assessment/12206270), direct file: https://ndownloader.figshare.com/files/24679091), containing `stain`/`unstain`/`segmentation` folders together — not separable, so the whole archive has to be pulled to get the `segmentation/images` + `segmentation/groundtruth` subfolders. No pretrained checkpoint was ever published by the authors (checked: figshare deposit is data-only, no `.pth`/`.h5` artifact; no author GitHub repo found) — this always requires training from scratch, on their labels or otherwise.
- **User is downloading the archive separately** (out of band, not run by this pipeline). Once extracted, drop the `segmentation/` folder at `datasets/DO_segmentation_gt/` (or update the path this phase's loader expects) with its `images`/`groundtruth` subfolders intact.
- **Deliverables:** `src/segmentation/unet_model.py` (architecture), `src/segmentation/unet_dataset.py` (loader for the `segmentation/images`+`groundtruth` pair, 512×512 patch extraction matching the paper's 20-sub-image split), `src/segmentation/train_unet.py`, `src/segmentation/segment_unet.py` (inference entry point mirroring `segment_classical()`'s signature so it's a drop-in alternative in Phase 1/2 call sites).
- **Validation plan:** reproduce the paper's own metric (mean IoU on a held-out split, normalized-input variant) as the primary sanity check that the reimplementation is faithful; then compare qualitatively against `classical_watershed.py` output on the same `datasets/DO` crops used in Phase 1/2 (periderm + parenchyma), and check inference speed on full-resolution sections since that's the practical motivation for a learned method (matches Phase 6.5's speed rationale).
- **Exit criteria:** U-Net reproduces mean IoU ≈0.70 on held-out data (paper's reported figure, normalized-input variant) or a documented explanation for any gap; visual QC on the same periderm/parenchyma crops shows boundary quality at least comparable to classical watershed, with no systematic failure mode (e.g. it should not reproduce or worsen the periderm under-detection issue noted in Phase 1); segmentation method becomes selectable (classical vs. U-Net) rather than classical being hard-coded, so downstream Phase 2+ code doesn't need to care which produced the instance masks.
- **Status:** not started — blocked on the archive download landing locally.

---

## Phase 7 (stretch) — Analysis tooling / interface
**Goal:** make the framework usable beyond a research script.

- CLI or notebook interface: image in → segmentation, feature table, graph visualization, latent embedding out.
- Batch comparison tooling (e.g. compare composition latents across species/conditions).
- Optional: simple web/artifact viewer for generated structures vs. real ones.

---

## Environment notes
- All phases run inside the `py311` Anaconda environment as the base interpreter — no venv/poetry/separate conda env should be introduced.
- Dependencies are declared in `pyproject.toml` and installed/resolved with `uv` (`uv sync`), not raw `pip`; day-to-day commands go through `uv run ...`.
- Any scripts, notebooks, or CI-like run instructions added in later phases should assume `conda activate py311` has already been run, followed by `uv run <cmd>` (or `conda run -n py311 uv run <cmd>` in automation contexts).
- All `torch` device selection uses the MPS backend on this Mac (`torch.device("mps" if torch.backends.mps.is_available() else "cpu")`), with `PYTORCH_ENABLE_MPS_FALLBACK=1` set to handle any `torch_geometric` ops without an MPS kernel yet.

## Sequencing notes
- Phases 1–2 are prerequisites for everything else and should be solid before any model training starts — feature/graph correctness bugs are much easier to catch by eye now than after they've propagated into a trained model.
- Phase 4 is a deliberate checkpoint: don't let field-representation bugs get blamed on the diffusion model.
- Phase 6 (data scale-up) can start in parallel with Phases 3–5 prototyping — sourcing more specimens is the long-lead-time item, so kick it off early rather than after hitting the data ceiling.
