"""Builds the Phase 2 graph for a single label map, for driving
build_all_graphs.py's per-image work as parallel OS processes (e.g.
`xargs -P`) instead of one sequential loop -- see
slurm/phase2_build_graphs.slurm. Same skip-if-exists/output-path convention
as build_all_graphs.py, so both are safe to mix/resume against the same
outputs/phase2/ directories.

Usage: uv run python -m src.graph.build_one_graph <path/to/name_full_labels.npy> [raw|norm]
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.build_graph import build_graph


def main(label_path: Path, variant: str = "raw"):
    if variant not in ("raw", "norm"):
        raise SystemExit(f"unknown variant {variant!r}, expected 'raw' or 'norm'")
    out_dir = out_dir_for(variant)
    name = label_path.stem.removesuffix("_full_labels")
    out_path = out_dir / f"{name}.pt"
    if out_path.exists():
        print(f"{name}: already cached, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        labels = np.load(label_path)
        t0 = time.time()
        data, node_df = build_graph(labels, normalize_by_size=(variant == "norm"))
        dt = time.time() - t0
        torch.save({"data": data, "node_df": node_df}, out_path)
        print(f"{name}: {data.num_nodes} nodes, {data.num_edges} edges, "
              f"{dt:.1f}s -> {out_path.name}")
    except Exception as e:
        print(f"{name}: FAILED - {e}", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    variant = sys.argv[2] if len(sys.argv) > 2 else "raw"
    main(Path(sys.argv[1]), variant)
