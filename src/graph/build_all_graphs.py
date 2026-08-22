"""Batch-build and cache the Phase 2 graph (torch_geometric Data + node
feature DataFrame) for every segmented image in outputs/phase1/, so Phase 3
batch training doesn't repeat the (non-trivial, ~10-30s/image) RAG
construction on every run. Skips images already cached. Mirrors
segment_all.py's progress-logging pattern.

Usage: uv run python -m src.graph.build_all_graphs [raw|norm]
  raw  (default) -- node features in raw pixel units, outputs/phase2/graphs/
  norm -- node features normalized by per-image size (see build_graph.py's
          normalize_by_size), outputs/phase2/graphs_norm/ -- the practical
          fix for the raw-pixel-resolution leak found in
          diagnose_image_variance.py, since no real physical (um)
          calibration exists for any of these datasets (see PLAN.md)
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_graph import build_graph

LABELS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase1"
PHASE2_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase2"


def out_dir_for(variant: str) -> Path:
    return PHASE2_DIR / ("graphs_norm" if variant == "norm" else "graphs")


def log(msg: str, log_path: Path):
    print(msg, flush=True)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "raw"
    if variant not in ("raw", "norm"):
        raise SystemExit(f"unknown variant {variant!r}, expected 'raw' or 'norm'")
    out_dir = out_dir_for(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_all_graphs_log.txt"

    label_paths = sorted(LABELS_DIR.glob("*_full_labels.npy"))
    log(f"\n=== build_all_graphs run start (variant={variant}), "
        f"{len(label_paths)} label maps found ===", log_path)

    done, skipped, failed = 0, 0, 0
    for i, label_path in enumerate(label_paths):
        name = label_path.stem.removesuffix("_full_labels")
        out_path = out_dir / f"{name}.pt"
        if out_path.exists():
            skipped += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: already cached, skipping", log_path)
            continue

        try:
            labels = np.load(label_path)
            t0 = time.time()
            data, node_df = build_graph(labels, normalize_by_size=(variant == "norm"))
            dt = time.time() - t0
            torch.save({"data": data, "node_df": node_df}, out_path)
            done += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: {data.num_nodes} nodes, "
                f"{data.num_edges} edges, {dt:.1f}s -> {out_path.name}", log_path)
        except Exception as e:
            failed += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: FAILED - {e}", log_path)
            log(traceback.format_exc(), log_path)

    log(f"=== build_all_graphs run complete (variant={variant}): {done} newly built, "
        f"{skipped} skipped, {failed} failed (of {len(label_paths)} total) ===", log_path)


if __name__ == "__main__":
    main()
