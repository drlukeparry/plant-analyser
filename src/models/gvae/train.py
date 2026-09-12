"""Phase 3 training loop: Graph-VAE on subgraphs sampled from the Phase 2 cell
graph(s). Supports both single-image training (original Phase 3 first-pass
mode) and multi-image batch training over cached graphs from
build_all_graphs.py (SubgraphSampler already accepted a list of graphs; this
just wires that up at the training-script level with global, not per-image,
feature standardization -- pooling stats across images is required for the
per-node feature distributions to be comparable when subgraphs are sampled
across a mix of images with different cell-size/density distributions).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.build_graph import build_graph
from src.models.gvae.model import GraphVAE
from src.models.gvae.sampling import SubgraphSampler

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
GRAPH_CACHE_DIR = out_dir_for("raw")  # kept for backward-compat imports (eval.py etc.)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def standardize(data, mean, std):
    data.x = (data.x - mean) / std
    return data


def _run_training_loop(
    graphs: list,
    in_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    beta: float,
    num_hops: int,
    max_nodes: int,
    dev,
):
    sampler = SubgraphSampler(graphs, num_hops=num_hops, max_nodes=max_nodes)
    model = GraphVAE(in_dim=in_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for step in range(steps):
        batch = sampler.sample_batch(batch_size).to(dev)
        opt.zero_grad()

        node_recon, edge_embed, mu, logvar = model(batch.x, batch.edge_index, batch.batch)

        node_loss = F.mse_loss(node_recon, batch.x)

        neg_edge_index = negative_sampling(
            batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=batch.edge_index.size(1)
        )
        pos_logits = (edge_embed[batch.edge_index[0]] * edge_embed[batch.edge_index[1]]).sum(-1)
        neg_logits = (edge_embed[neg_edge_index[0]] * edge_embed[neg_edge_index[1]]).sum(-1)
        edge_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        edge_logits = torch.cat([pos_logits, neg_logits])
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = node_loss + edge_loss + beta * kl
        loss.backward()
        opt.step()

        history.append((step, loss.item(), node_loss.item(), edge_loss.item(), kl.item()))
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d} loss={loss.item():.4f} node={node_loss.item():.4f} "
                  f"edge={edge_loss.item():.4f} kl={kl.item():.4f}")

    return model, history


def train(
    labels_path: str,
    steps: int = 800,
    batch_size: int = 16,
    lr: float = 1e-3,
    beta: float = 0.01,
    num_hops: int = 3,
    max_nodes: int = 300,
):
    """Single-image training (original Phase 3 first-pass mode)."""
    dev = device()
    labels = np.load(labels_path)
    print(f"building graph from {labels_path} ...")
    full_data, node_df = build_graph(labels)
    print(f"full graph: {full_data.num_nodes} nodes, {full_data.num_edges} edges")

    mean = full_data.x.mean(dim=0, keepdim=True)
    std = full_data.x.std(dim=0, keepdim=True).clamp(min=1e-6)
    np.savez(OUT_DIR / "feature_scaler.npz", mean=mean.numpy(), std=std.numpy())
    full_data.x = (full_data.x - mean) / std

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, history = _run_training_loop(
        [full_data], full_data.x.size(1), steps, batch_size, lr, beta, num_hops, max_nodes, dev
    )

    torch.save(model.state_dict(), OUT_DIR / "gvae.pt")
    np.save(OUT_DIR / "train_history.npy", np.array(history))
    print(f"saved model + history to {OUT_DIR}")
    return model, full_data, node_df, mean, std


def train_multi(
    graph_paths: list[Path],
    steps: int = 1500,
    batch_size: int = 16,
    lr: float = 1e-3,
    beta: float = 0.01,
    num_hops: int = 3,
    max_nodes: int = 300,
    tag: str = "multi",
):
    """Batch training over cached per-image graphs (see build_all_graphs.py).
    Loads all graphs up front (their combined size is small -- feature
    matrices only, no pixel data) and standardizes with statistics pooled
    across every image, not per-image, so subgraphs sampled from different
    source images land in the same feature space.
    """
    dev = device()
    print(f"loading {len(graph_paths)} cached graphs ...")
    graphs = []
    for p in graph_paths:
        cached = torch.load(p, weights_only=False)
        graphs.append(cached["data"])
    total_nodes = sum(g.num_nodes for g in graphs)
    total_edges = sum(g.num_edges for g in graphs)
    print(f"loaded {len(graphs)} graphs: {total_nodes} total nodes, {total_edges} total edges")

    all_x = torch.cat([g.x for g in graphs], dim=0)
    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True).clamp(min=1e-6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"feature_scaler_{tag}.npz", mean=mean.numpy(), std=std.numpy())

    for g in graphs:
        g.x = (g.x - mean) / std

    model, history = _run_training_loop(
        graphs, graphs[0].x.size(1), steps, batch_size, lr, beta, num_hops, max_nodes, dev
    )

    torch.save(model.state_dict(), OUT_DIR / f"gvae_{tag}.pt")
    np.save(OUT_DIR / f"train_history_{tag}.npy", np.array(history))
    print(f"saved model + history to {OUT_DIR} (tag={tag})")
    return model, graphs, mean, std


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".npy"):
        # Single-image mode, backward compatible with the Phase 3 first pass.
        train(sys.argv[1])
    else:
        # Multi-image batch mode: train over cached graphs in
        # outputs/phase2/graphs[_norm]/ (see build_all_graphs.py). Remaining
        # args (any order): "norm" selects the size-normalized graph cache
        # (see build_graph.py's normalize_by_size) instead of raw pixel
        # units; a dataset-prefix arg (e.g. "DO") restricts to that dataset
        # only. Both tag the checkpoint accordingly.
        args = sys.argv[1:]
        variant = "norm" if "norm" in args else "raw"
        prefix = next((a for a in args if a != "norm"), None)
        graph_dir = out_dir_for(variant)
        pattern = f"{prefix}_*.pt" if prefix else "*.pt"
        tag = "_".join(t for t in (prefix, variant if variant == "norm" else None) if t) or "multi"
        graph_paths = sorted(graph_dir.glob(pattern))
        if not graph_paths:
            raise FileNotFoundError(
                f"no cached graphs matching {pattern!r} found in {graph_dir} -- run "
                f"`uv run python -m src.graph.build_all_graphs {variant}` first"
            )
        train_multi(graph_paths, tag=tag)
