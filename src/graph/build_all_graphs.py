"""Batch-build and cache the Phase 2 graph (torch_geometric Data + node
feature DataFrame) for every segmented image in outputs/phase1/, so Phase 3
batch training doesn't repeat the (non-trivial, ~10-30s/image) RAG
construction on every run. Skips images already cached. Mirrors
segment_all.py's progress-logging pattern.

Usage: uv run python -m src.graph.build_all_graphs
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
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "phase2" / "graphs"
LOG_PATH = OUT_DIR / "build_all_graphs_log.txt"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(LABELS_DIR.glob("*_full_labels.npy"))
    log(f"\n=== build_all_graphs run start, {len(label_paths)} label maps found ===")

    done, skipped, failed = 0, 0, 0
    for i, label_path in enumerate(label_paths):
        name = label_path.stem.removesuffix("_full_labels")
        out_path = OUT_DIR / f"{name}.pt"
        if out_path.exists():
            skipped += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: already cached, skipping")
            continue

        try:
            labels = np.load(label_path)
            t0 = time.time()
            data, node_df = build_graph(labels)
            dt = time.time() - t0
            torch.save({"data": data, "node_df": node_df}, out_path)
            done += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: {data.num_nodes} nodes, "
                f"{data.num_edges} edges, {dt:.1f}s -> {out_path.name}")
        except Exception as e:
            failed += 1
            log(f"[{i+1}/{len(label_paths)}] {name}: FAILED - {e}")
            log(traceback.format_exc())

    log(f"=== build_all_graphs run complete: {done} newly built, {skipped} skipped, "
        f"{failed} failed (of {len(label_paths)} total) ===")


if __name__ == "__main__":
    main()
