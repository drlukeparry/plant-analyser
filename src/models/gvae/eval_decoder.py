"""ROADMAP2 R1: quantify the (currently unused-downstream) GVAE decoder's
reconstruction quality -- node feature + edge reconstruction -- since
PLAN.md's Phase 3 flagged this as never measured beyond the training loss
curve, and R2 (graph-space generation) would build directly on this decoder.

Caveat, stated plainly: `gvae_norm.pt` was trained on subgraphs sampled from
every image in outputs/phase2/graphs_norm/ (see train.py::train_multi) --
there is no held-out image split from training. This script therefore
measures in-sample reconstruction quality (same distribution the model was
fit on), not generalization to unseen images. That's still a real, useful
number (does the decoder reconstruct what it was trained to reconstruct at
all?) but should not be read as a generalization test.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.build_graph import NODE_FEATURE_COLS_PX, _PX_TO_NORM_COL
from src.graph.build_all_graphs import out_dir_for
from src.models.gvae.model import GraphVAE
from src.models.gvae.sampling import SubgraphSampler
from src.models.gvae.train import device

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase3"
FEATURE_NAMES = [_PX_TO_NORM_COL.get(c, c) for c in NODE_FEATURE_COLS_PX]


def load_model_and_graphs(tag: str = "norm"):
    dev = device()
    scaler = np.load(OUT_DIR / f"feature_scaler_{tag}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])

    graph_dir = out_dir_for(tag)
    graph_paths = sorted(graph_dir.glob("*.pt"))
    graphs = [torch.load(p, weights_only=False)["data"] for p in graph_paths]
    for g in graphs:
        g.x = (g.x - mean) / std

    model = GraphVAE(in_dim=graphs[0].x.size(1)).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / f"gvae_{tag}.pt", map_location=dev))
    model.eval()
    return model, graphs, mean, std, dev


@torch.no_grad()
def evaluate(n_subgraphs: int = 300, num_hops: int = 3, max_nodes: int = 300, tag: str = "norm", seed: int = 0):
    torch.manual_seed(seed)
    model, graphs, mean, std, dev = load_model_and_graphs(tag)
    sampler = SubgraphSampler(graphs, num_hops=num_hops, max_nodes=max_nodes)

    real_all, recon_all = [], []
    edge_logits_all, edge_labels_all = [], []
    n_nodes_total, n_subgraphs_used = 0, 0

    from torch_geometric.data import Batch
    from torch_geometric.utils import negative_sampling

    batch_size = 16
    n_batches = (n_subgraphs + batch_size - 1) // batch_size
    for _ in range(n_batches):
        subs = [sampler.sample() for _ in range(batch_size)]
        subs = [s for s in subs if s.num_nodes >= 8]
        if not subs:
            continue
        batch = Batch.from_data_list(subs).to(dev)

        mu, logvar = model.encode(batch.x, batch.edge_index, batch.batch)
        node_recon, edge_embed = model.decode(mu, batch.edge_index, batch.batch)

        real_all.append(batch.x.cpu())
        recon_all.append(node_recon.cpu())
        n_nodes_total += batch.num_nodes
        n_subgraphs_used += len(subs)

        neg_edge_index = negative_sampling(
            batch.edge_index, num_nodes=batch.num_nodes, num_neg_samples=batch.edge_index.size(1)
        )
        pos_logits = (edge_embed[batch.edge_index[0]] * edge_embed[batch.edge_index[1]]).sum(-1)
        neg_logits = (edge_embed[neg_edge_index[0]] * edge_embed[neg_edge_index[1]]).sum(-1)
        edge_logits_all.append(torch.cat([pos_logits, neg_logits]).cpu())
        edge_labels_all.append(torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]).cpu())

    real = torch.cat(real_all, dim=0).numpy()
    recon = torch.cat(recon_all, dim=0).numpy()
    edge_logits = torch.cat(edge_logits_all, dim=0).numpy()
    edge_labels = torch.cat(edge_labels_all, dim=0).numpy()

    print(f"evaluated {n_subgraphs_used} subgraphs, {n_nodes_total} nodes total, "
          f"{len(edge_labels)} edge predictions (half positive, half negative)")

    # --- Node feature reconstruction, per feature, in standardized units ---
    print("\nnode feature reconstruction (standardized units):")
    print(f"{'feature':<28} {'MSE':>8} {'corr(real,recon)':>18}")
    per_feature_mse, per_feature_corr = {}, {}
    for i, name in enumerate(FEATURE_NAMES):
        mse = float(np.mean((real[:, i] - recon[:, i]) ** 2))
        corr = float(np.corrcoef(real[:, i], recon[:, i])[0, 1]) if np.std(recon[:, i]) > 1e-8 else float("nan")
        per_feature_mse[name] = mse
        per_feature_corr[name] = corr
        print(f"{name:<28} {mse:>8.3f} {corr:>18.3f}")

    overall_mse = float(np.mean((real - recon) ** 2))
    print(f"\noverall node MSE (standardized units, mean over all features): {overall_mse:.4f}")
    print("(for reference: a decoder that just predicts the per-feature training mean -> "
          "standardized MSE = 1.0 per feature, since features are z-scored to unit variance; "
          "values well below 1.0 indicate real reconstruction, not just predicting the mean)")

    # --- Edge (adjacency) reconstruction ---
    pred = edge_logits > 0
    tp = float(np.sum(pred & (edge_labels == 1)))
    fp = float(np.sum(pred & (edge_labels == 0)))
    fn = float(np.sum((~pred) & (edge_labels == 1)))
    tn = float(np.sum((~pred) & (edge_labels == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / len(edge_labels)
    print(f"\nedge reconstruction (real adjacency vs. negative-sampled non-edges, threshold logit>0):")
    print(f"  accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    # --- Plot: real vs. recon scatter for a few interpretable features ---
    plot_feats = ["radial_distance_norm", "area_norm", "orientation", "elongation"]
    plot_feats = [f for f in plot_feats if f in FEATURE_NAMES]
    fig, axes = plt.subplots(1, len(plot_feats), figsize=(5 * len(plot_feats), 5))
    if len(plot_feats) == 1:
        axes = [axes]
    rng = np.random.default_rng(seed)
    subsample = rng.choice(len(real), size=min(3000, len(real)), replace=False)
    for ax, name in zip(axes, plot_feats):
        i = FEATURE_NAMES.index(name)
        ax.scatter(real[subsample, i], recon[subsample, i], s=3, alpha=0.15)
        lo, hi = real[subsample, i].min(), real[subsample, i].max()
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x (perfect)")
        ax.set_xlabel("real (standardized)")
        ax.set_ylabel("reconstructed (standardized)")
        ax.set_title(f"{name}\ncorr={per_feature_corr[name]:.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / f"decoder_eval_{tag}.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved {out_path}")

    np.savez(
        OUT_DIR / f"decoder_eval_{tag}.npz",
        feature_names=np.array(FEATURE_NAMES),
        per_feature_mse=np.array([per_feature_mse[n] for n in FEATURE_NAMES]),
        per_feature_corr=np.array([per_feature_corr[n] for n in FEATURE_NAMES]),
        overall_node_mse=overall_mse,
        edge_accuracy=accuracy, edge_precision=precision, edge_recall=recall, edge_f1=f1,
    )
    return per_feature_mse, per_feature_corr, overall_mse, dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1
    )


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "norm"
    evaluate(tag=tag)
