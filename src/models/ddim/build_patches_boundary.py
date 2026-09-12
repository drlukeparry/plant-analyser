"""SDF-crop conditioning experiment (see ROADMAP2/PLAN.md discussion):
DO-only, native-resolution 256px patches, each paired with a spatial crop of
the whole-image `boundary_sdf` (build_boundary_sdf.py) stacked on as a 4th
channel -- everything else (GVAE-latent conditioning, normalize_by_size
target field) kept identical to build_patches.py so this isolates one
variable: does giving the model a real positional signal change what it
generates, versus the existing patch pipeline which has none.

(A gradient-of-boundary_sdf conditioning channel was tried and removed --
it introduces real medial-axis discontinuities that are harder to learn
from, and its hypothesized benefit for orientation alignment was never
validated before being dropped.)

Separate script from build_patches.py (rather than parameterizing it) so the
existing Phase 5/6 patch set/checkpoints stay reproducible and untouched.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.field import rasterize_field
from src.models.ddim.build_patches import sample_patch_conditioning
from src.models.gvae.model import GraphVAE
from src.models.gvae.train import device

LABELS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase1"
PHASE3_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"
BOUNDARY_SDF_DIR = OUT_DIR / "boundary_sdf"
LOG_PATH = OUT_DIR / "build_patches_boundary_log.txt"

PATCH_SIZE = 256
PATCHES_PER_IMAGE = 32
MIN_INTERIOR_FRAC = 0.02  # was 0.15 -- excluded real background/edge-adjacent crops from
# training entirely (measured: generated cell-count range 38-55 vs. real 7-71), leaving the
# model with no background concept. MIN_NODES below still rejects near-empty crops.
MIN_NODES = 8
BG_CLIP = 15.0  # was rasterize_field's default 50.0 -- ~4x a DO cell radius (median diameter
# 12.7px), so the sdf channel's variance was dominated by background/wall distance rather than
# the cell-interior detail tessellation actually needs (measured: outside-cell std 4.4x interior)


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def main(species: str = "DO", tag: str = "norm"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()

    scaler = np.load(PHASE3_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])

    graph_paths = sorted(out_dir_for(tag).glob(f"{species}_*.pt"))
    print(f"{len(graph_paths)} {species} images")

    probe = torch.load(graph_paths[0], weights_only=False)
    model = GraphVAE(in_dim=probe["data"].x.size(1)).to(dev)
    model.load_state_dict(torch.load(PHASE3_DIR / f"gvae_{tag}.pt", map_location=dev))
    model.eval()

    rng = np.random.default_rng(0)
    all_fields, all_conds, all_sources, all_locations = [], [], [], []

    log(f"\n=== build_patches_boundary run start, {len(graph_paths)} {species} images, "
        f"{PATCHES_PER_IMAGE} patches/image, patch_size={PATCH_SIZE} ===")

    for i, gp in enumerate(graph_paths):
        name = gp.stem
        boundary_path = BOUNDARY_SDF_DIR / f"{name}.npy"
        if not boundary_path.exists():
            log(f"[{i+1}/{len(graph_paths)}] {name}: SKIPPED - no boundary_sdf cached")
            continue
        try:
            cached = torch.load(gp, weights_only=False)
            data, node_df = cached["data"], cached["node_df"]
            labels = np.load(LABELS_DIR / f"{name}_full_labels.npy")
            boundary_sdf = np.load(boundary_path)
            assert boundary_sdf.shape == labels.shape, \
                f"{name}: boundary_sdf {boundary_sdf.shape} != labels {labels.shape}"

            t0 = time.time()
            field = rasterize_field(labels, node_df, normalize_by_size=True, bg_clip=BG_CLIP)
            h, w = labels.shape

            n_taken, n_tried = 0, 0
            while n_taken < PATCHES_PER_IMAGE and n_tried < PATCHES_PER_IMAGE * 8:
                n_tried += 1
                r0 = int(rng.integers(0, h - PATCH_SIZE))
                c0 = int(rng.integers(0, w - PATCH_SIZE))
                patch_field = field[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
                interior_frac = (patch_field[..., 0] > 0).mean()
                if interior_frac < MIN_INTERIOR_FRAC:
                    continue
                cond = sample_patch_conditioning(data, node_df, r0, c0, PATCH_SIZE, model, mean, std, dev)
                if cond is None:
                    continue
                boundary_crop = boundary_sdf[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
                patch_4ch = np.concatenate([patch_field, boundary_crop[..., None]], axis=-1)
                all_fields.append(patch_4ch.astype(np.float32))
                all_conds.append(cond.astype(np.float32))
                all_sources.append(name)
                all_locations.append((r0, c0))
                n_taken += 1

            log(f"[{i+1}/{len(graph_paths)}] {name}: {n_taken}/{PATCHES_PER_IMAGE} patches "
                f"({n_tried} tried), rasterize {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"[{i+1}/{len(graph_paths)}] {name}: FAILED - {e}")

    fields_arr = np.stack(all_fields, axis=0)  # (N, 256, 256, 4): 3 target + boundary_sdf
    conds_arr = np.stack(all_conds, axis=0)    # (N, latent_dim)
    np.save(OUT_DIR / "patch_fields_boundary.npy", fields_arr)
    np.save(OUT_DIR / "patch_conds_boundary.npy", conds_arr)
    with open(OUT_DIR / "patch_sources_boundary.txt", "w") as f:
        f.write("\n".join(all_sources))
    np.save(OUT_DIR / "patch_locations_boundary.npy", np.array(all_locations, dtype=np.int64))

    log(f"=== build_patches_boundary run complete: {len(all_fields)} patches from "
        f"{len(graph_paths)} {species} images -> {OUT_DIR}/patch_fields_boundary.npy "
        f"{fields_arr.shape}, patch_conds_boundary.npy {conds_arr.shape} ===")


if __name__ == "__main__":
    species = sys.argv[1] if len(sys.argv) > 1 else "DO"
    main(species)
