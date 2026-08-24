"""Phase 5 dataset prep: crop size-normalized fields (Phase 4) into patches
and pair each with a Phase 3 GVAE composition latent (encoded from that
patch's own local subgraph), for conditional DDIM training.

Uses the same normalize_by_size=True basis as Phase 3's fix (see PLAN.md) so
patches drawn from images at very different native resolutions land in a
comparable feature/field space -- rasterizing the raw pixel-unit field here
would reintroduce the exact resolution-leak bug Phase 3 spent significant
effort diagnosing and fixing, one layer downstream.
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
from src.models.gvae.model import GraphVAE
from src.models.gvae.train import device

LABELS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase1"
PHASE3_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"
LOG_PATH = OUT_DIR / "build_patches_log.txt"

PATCH_SIZE = 128
PATCHES_PER_IMAGE = 16
MIN_INTERIOR_FRAC = 0.15  # skip patches that are mostly background
MIN_NODES = 8


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def sample_patch_conditioning(data, node_df, r0, c0, patch_size, model, mean, std, dev):
    """Encode the subgraph of cells whose centroid falls inside the patch
    window into a composition latent, using the same trained encoder Phase 3
    validated -- the conditioning signal for DDIM is literally "what did the
    Phase 3 model think this local tissue looked like."""
    m = (
        (node_df["centroid_row"].values >= r0) & (node_df["centroid_row"].values < r0 + patch_size)
        & (node_df["centroid_col"].values >= c0) & (node_df["centroid_col"].values < c0 + patch_size)
    )
    node_idx = torch.tensor(np.nonzero(m)[0], dtype=torch.long)
    if node_idx.numel() < MIN_NODES:
        return None
    edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    x = (data.x[node_idx] - mean) / std
    with torch.no_grad():
        batch = torch.zeros(x.size(0), dtype=torch.long)
        mu, _ = model.encode(x.to(dev), edge_index.to(dev), batch.to(dev))
    return mu.cpu().numpy()[0]


def main(n_images: int | None = None, tag: str = "norm"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()

    scaler = np.load(PHASE3_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])

    graph_paths = sorted(out_dir_for(tag).glob("*.pt"))
    if n_images is not None:
        graph_paths = graph_paths[:n_images]

    probe = torch.load(graph_paths[0], weights_only=False)
    model = GraphVAE(in_dim=probe["data"].x.size(1)).to(dev)
    model.load_state_dict(torch.load(PHASE3_DIR / f"gvae_{tag}.pt", map_location=dev))
    model.eval()

    rng = np.random.default_rng(0)
    all_fields, all_conds, all_sources = [], [], []

    log(f"\n=== build_patches run start, {len(graph_paths)} images, "
        f"{PATCHES_PER_IMAGE} patches/image, patch_size={PATCH_SIZE} ===")

    for i, gp in enumerate(graph_paths):
        name = gp.stem
        try:
            cached = torch.load(gp, weights_only=False)
            data, node_df = cached["data"], cached["node_df"]
            labels = np.load(LABELS_DIR / f"{name}_full_labels.npy")

            t0 = time.time()
            field = rasterize_field(labels, node_df, normalize_by_size=True)
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
                all_fields.append(patch_field.astype(np.float32))
                all_conds.append(cond.astype(np.float32))
                all_sources.append(name)
                n_taken += 1

            log(f"[{i+1}/{len(graph_paths)}] {name}: {n_taken}/{PATCHES_PER_IMAGE} patches "
                f"({n_tried} tried), rasterize {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"[{i+1}/{len(graph_paths)}] {name}: FAILED - {e}")

    fields_arr = np.stack(all_fields, axis=0)  # (N, patch, patch, 4)
    conds_arr = np.stack(all_conds, axis=0)    # (N, latent_dim)
    np.save(OUT_DIR / "patch_fields.npy", fields_arr)
    np.save(OUT_DIR / "patch_conds.npy", conds_arr)
    with open(OUT_DIR / "patch_sources.txt", "w") as f:
        f.write("\n".join(all_sources))

    log(f"=== build_patches run complete: {len(all_fields)} patches from "
        f"{len(graph_paths)} images -> {OUT_DIR}/patch_fields.npy "
        f"{fields_arr.shape}, patch_conds.npy {conds_arr.shape} ===")


if __name__ == "__main__":
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_images)
