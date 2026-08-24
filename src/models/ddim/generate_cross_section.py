"""Adapts the Phase 5 patch-level DDIM into a full stem cross-section: tiles
many independently-generated 128px patches into one large canvas, using a
real reference image as a spatial/conditioning template so the overall disc
shape and radial zonation (small cells near one edge, large near the other)
come from real structure, while the cell-level content within each tile is
genuinely generated, not copied from the reference.

This is explicitly beyond PLAN.md's current Phase 5 scope (real
whole-structure generation is Phase 6+, once more data/training exists) --
a demonstration of how far the existing patch-conditioned model can be
pushed, not a claim that the model learned global tissue shape. Two real
limitations follow directly from that: (1) the model was only ever trained
on patches with substantial interior tissue, so it has no notion of "this
region is background" -- the overall silhouette here is copied from the
reference image's real tissue mask, not generated; (2) adjacent tiles are
generated independently with no cross-tile consistency, so seams between
tiles are expected (no overlap-blending implemented).
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from PIL import Image

from src.graph.build_all_graphs import out_dir_for
from src.graph.features import extract_cell_features
from src.graph.tessellate import tessellate_from_field
from src.models.ddim.build_patches import PATCH_SIZE, sample_patch_conditioning
from src.models.ddim.sample import generate, load_model
from src.models.gvae.model import GraphVAE
from src.models.gvae.train import device
from src.segmentation.classical_watershed import tissue_mask as compute_tissue_mask

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"
PHASE3_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
LABELS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase1"
DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
MIN_TISSUE_MASK_FRAC = 0.3  # grid-cell-level silhouette check (coarse, holes-filled)


def load_gvae(dev, tag: str = "norm"):
    """The conditioning encoder (Phase 3) -- separate from the DDIM UNet
    (Phase 5) loaded by sample.py's load_model(); both are needed here."""
    scaler = np.load(PHASE3_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    probe = torch.load(next(out_dir_for(tag).glob("*.pt")), weights_only=False)
    gvae = GraphVAE(in_dim=probe["data"].x.size(1)).to(dev)
    gvae.load_state_dict(torch.load(PHASE3_DIR / f"gvae_{tag}.pt", map_location=dev))
    gvae.eval()
    return gvae, mean, std


def main(ref_name: str = "DO_0000", grid_n: int = 10, seed: int = 0):
    ddim_model, field_mean, field_std, dev = load_model()
    gvae_model, gvae_mean, gvae_std = load_gvae(dev)

    cached = torch.load(out_dir_for("norm") / f"{ref_name}.pt", weights_only=False)
    data, node_df = cached["data"], cached["node_df"]
    ref_labels = np.load(LABELS_DIR / f"{ref_name}_full_labels.npy")
    h, w = ref_labels.shape

    # Coarse, holes-filled tissue silhouette (same function segmentation
    # itself uses to exclude background) -- checked per grid cell over each
    # cell's full real-space footprint, not a literal 128px window at the
    # sample point. A raw per-pixel label>0 check at just the 128px sample
    # location is fooled by internal lumens/thin gaps (a small window can
    # land on a real hole even deep inside solid tissue), producing a
    # scattered, shapeless mask instead of the true disc/lobed outline.
    species_code = ref_name.split("_")[0]
    ref_rgb = np.array(Image.open(DATASETS_DIR / species_code / "inputimages" / f"{ref_name}.jpg").convert("RGB"))
    silhouette = compute_tissue_mask(ref_rgb)

    tissue_rows, tissue_cols = np.nonzero(ref_labels > 0)
    r_min, r_max = tissue_rows.min(), tissue_rows.max()
    c_min, c_max = tissue_cols.min(), tissue_cols.max()
    cell_h = max((r_max - r_min) / grid_n, PATCH_SIZE)
    cell_w = max((c_max - c_min) / grid_n, PATCH_SIZE)
    print(f"reference {ref_name}: {h}x{w} full image, "
          f"tissue bbox {r_max-r_min}x{c_max-c_min}, {len(node_df)} real cells")

    canvas_size = grid_n * PATCH_SIZE
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.float32)
    tissue_mask = np.zeros((canvas_size, canvas_size), dtype=bool)

    rng = np.random.default_rng(seed)
    n_generated, n_skipped = 0, 0
    t0 = time.time()
    for i in range(grid_n):
        for j in range(grid_n):
            # proportional mapping: grid cell (i,j) -> the corresponding
            # region within the tissue's bounding box, preserving its
            # overall layout without the surrounding background margin.
            # r0/c0 (the actual 128px conditioning/generation window) are
            # centered in this block, not at its corner -- corners are much
            # more likely to straddle a tissue boundary or land on a locally
            # sparse/lumen region even when the block as a whole is solid
            # tissue (measured directly: at the block corner, 63 of 78
            # silhouette-passing blocks had too few real cells in their
            # specific 128px window to encode a conditioning latent at all;
            # centering fixes the mismatch between "this block is tissue"
            # and "this specific small window has real cells in it").
            block_r0 = int(r_min + i * cell_h)
            block_r1 = int(min(r_min + (i + 1) * cell_h, h))
            block_c0 = int(c_min + j * cell_w)
            block_c1 = int(min(c_min + (j + 1) * cell_w, w))

            # silhouette check over the full grid-cell footprint (real
            # pixels), not just the small 128px conditioning/generation
            # window -- this is what actually decides "is this part of the
            # tissue" for shape purposes.
            silhouette_frac = silhouette[block_r0:block_r1, block_c0:block_c1].mean()

            ri0, ri1 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            ci0, ci1 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
            if silhouette_frac < MIN_TISSUE_MASK_FRAC:
                n_skipped += 1
                continue

            # Even a block that's solidly tissue by the (coarse) silhouette
            # often has too few real cell centroids in one *specific* 128px
            # window to encode a conditioning latent (measured: 53/78
            # silhouette-passing blocks failed at their exact center, most
            # with 4-7 cells -- just under the min-8 threshold, not empty).
            # build_patches.py handled the analogous problem with up to 8x
            # random retries per patch; do the same here, searching within
            # this block so the result stays roughly in the right place.
            cond_vec = None
            for _try in range(8):
                if _try == 0:
                    r0 = min(max(int((block_r0 + block_r1) / 2 - PATCH_SIZE / 2), 0), h - PATCH_SIZE)
                    c0 = min(max(int((block_c0 + block_c1) / 2 - PATCH_SIZE / 2), 0), w - PATCH_SIZE)
                else:
                    r0 = int(rng.integers(block_r0, max(block_r1 - PATCH_SIZE, block_r0 + 1)))
                    c0 = int(rng.integers(block_c0, max(block_c1 - PATCH_SIZE, block_c0 + 1)))
                    r0 = min(max(r0, 0), h - PATCH_SIZE)
                    c0 = min(max(c0, 0), w - PATCH_SIZE)
                cond_vec = sample_patch_conditioning(
                    data, node_df, r0, c0, PATCH_SIZE, gvae_model, gvae_mean, gvae_std, dev
                )
                if cond_vec is not None:
                    break
            if cond_vec is None:
                n_skipped += 1
                continue

            cond = torch.tensor(cond_vec[None], dtype=torch.float32)
            field = generate(ddim_model, cond, field_mean, field_std, dev, seed=seed + i * grid_n + j)
            canvas[ri0:ri1, ci0:ci1] = field
            tissue_mask[ri0:ri1, ci0:ci1] = True
            n_generated += 1

        print(f"  row {i+1}/{grid_n} done ({time.time()-t0:.1f}s elapsed)")

    print(f"generated {n_generated} tiles, skipped {n_skipped} (background) "
          f"in {time.time()-t0:.1f}s")

    # zero out sdf outside the real tissue silhouette so the tessellation
    # doesn't treat blank canvas as more tissue (see module docstring: the
    # model has no background concept, so the silhouette comes from the
    # reference, not from the generation itself).
    canvas[..., 0][~tissue_mask] = -50.0

    recovered = tessellate_from_field(canvas)
    recovered_df = extract_cell_features(recovered)
    print(f"tessellated generated cross-section: {len(recovered_df)} cells")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(ref_labels, cmap="nipy_spectral")
    axes[0].set_title(f"reference: {ref_name} ({len(node_df)} real cells)")
    axes[1].imshow(canvas[..., 0], cmap="viridis")
    axes[1].set_title("generated field (sdf channel), tiled")
    axes[2].imshow(recovered, cmap="nipy_spectral")
    axes[2].set_title(f"tessellated generated cross-section ({len(recovered_df)} cells)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Phase 5 extension: full cross-section from tiled DDIM patches "
                 f"(grid {grid_n}x{grid_n}, {n_generated} generated / {n_skipped} background)", fontsize=12)
    fig.tight_layout()
    out_path = OUT_DIR / f"generated_cross_section_{ref_name}.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")

    np.save(OUT_DIR / f"generated_cross_section_{ref_name}_labels.npy", recovered)
    return recovered, recovered_df


if __name__ == "__main__":
    ref_name = sys.argv[1] if len(sys.argv) > 1 else "DO_0000"
    grid_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(ref_name, grid_n)
