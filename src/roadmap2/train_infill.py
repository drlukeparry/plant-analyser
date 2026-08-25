"""Train + evaluate the R2 node-infill model, with a with/without-control-
field ablation -- the direct test of ROADMAP2's core claim that explicit
size-gradient/alignment-flow control fields (extracted in control_fields.py,
sanity-checked in visualize_fields.py) are a useful conditioning signal for
generating/completing cell attributes, not just plausible-sounding.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_all_graphs import out_dir_for
from src.graph.build_graph import NODE_FEATURE_COLS_PX, _PX_TO_NORM_COL
from src.models.gvae.train import device
from src.roadmap2.control_fields import add_control_fields
from src.roadmap2.infill_model import NodeInfillGNN

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "roadmap2"
FEATURE_NAMES = [_PX_TO_NORM_COL.get(c, c) for c in NODE_FEATURE_COLS_PX]
CTRL_COLS = ["size_gradient_target", "_flow_sin", "_flow_cos", "alignment_flow_coherence"]


def load_graphs_with_ctrl(tag: str = "norm"):
    graph_dir = out_dir_for(tag)
    graph_paths = sorted(graph_dir.glob("*.pt"))
    graphs, ctrls = [], []
    for p in graph_paths:
        cached = torch.load(p, weights_only=False)
        node_df = add_control_fields(cached["node_df"])
        node_df["_flow_sin"] = np.sin(node_df["alignment_flow_angle"])
        node_df["_flow_cos"] = np.cos(node_df["alignment_flow_angle"])
        ctrl = torch.tensor(node_df[CTRL_COLS].values, dtype=torch.float32)
        graphs.append(cached["data"])
        ctrls.append(ctrl)
    return graphs, ctrls


def sample_subgraph_with_ctrl(data: Data, ctrl: torch.Tensor, num_hops: int = 3, max_nodes: int = 300):
    seed = torch.randint(0, data.num_nodes, (1,)).item()
    node_idx, edge_index, _, _ = k_hop_subgraph(
        seed, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    if node_idx.numel() > max_nodes:
        keep = torch.randperm(node_idx.numel())[:max_nodes]
        keep_set = set(keep.tolist())
        node_mask = torch.tensor([i in keep_set for i in range(node_idx.numel())])
        remap = -torch.ones(node_idx.numel(), dtype=torch.long)
        remap[node_mask] = torch.arange(node_mask.sum())
        edge_sel = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = remap[edge_index[:, edge_sel]]
        node_idx = node_idx[node_mask]
    x = data.x[node_idx]
    c = ctrl[node_idx]
    return Data(x=x, edge_index=edge_index, ctrl=c)


class Sampler:
    def __init__(self, graphs, ctrls, num_hops=3, max_nodes=300):
        self.graphs, self.ctrls = graphs, ctrls
        self.num_hops, self.max_nodes = num_hops, max_nodes

    def sample(self):
        i = torch.randint(0, len(self.graphs), (1,)).item()
        return sample_subgraph_with_ctrl(self.graphs[i], self.ctrls[i], self.num_hops, self.max_nodes)

    def sample_batch(self, batch_size):
        subs = [self.sample() for _ in range(batch_size)]
        subs = [s for s in subs if s.num_nodes >= 8]
        return Batch.from_data_list(subs)


def mask_batch(batch, mask_frac: float, dev):
    n = batch.num_nodes
    mask = (torch.rand(n, device=dev) < mask_frac).float()
    x_masked = batch.x * (1 - mask.unsqueeze(-1))
    return x_masked, mask


def train_one(use_ctrl: bool, graphs, ctrls, mean, std, steps: int, batch_size: int,
              mask_frac: float, dev, seed: int = 0):
    torch.manual_seed(seed)
    sampler = Sampler(graphs, ctrls)
    attr_dim = graphs[0].x.size(1)
    ctrl_dim = ctrls[0].size(1) if use_ctrl else ctrls[0].size(1)  # same input width either way
    model = NodeInfillGNN(attr_dim=attr_dim, ctrl_dim=ctrl_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    for step in range(steps):
        batch = sampler.sample_batch(batch_size).to(dev)
        x_masked, mask = mask_batch(batch, mask_frac, dev)
        ctrl = batch.ctrl if use_ctrl else torch.zeros_like(batch.ctrl)

        opt.zero_grad()
        pred = model(x_masked, mask, ctrl, batch.edge_index)
        masked_idx = mask.bool()
        if masked_idx.sum() == 0:
            continue
        loss = F.mse_loss(pred[masked_idx], batch.x[masked_idx])
        loss.backward()
        opt.step()
        history.append(loss.item())
        if step % 200 == 0 or step == steps - 1:
            print(f"  [{'ctrl' if use_ctrl else 'no_ctrl'}] step {step:4d} loss={loss.item():.4f}")

    return model, history


@torch.no_grad()
def evaluate(model, use_ctrl: bool, graphs, ctrls, dev, n_subgraphs=300, mask_frac=0.3, seed=1):
    torch.manual_seed(seed)
    sampler = Sampler(graphs, ctrls)
    model.eval()
    real_all, pred_all = [], []
    batch_size = 16
    n_batches = (n_subgraphs + batch_size - 1) // batch_size
    for _ in range(n_batches):
        batch = sampler.sample_batch(batch_size).to(dev)
        x_masked, mask = mask_batch(batch, mask_frac, dev)
        ctrl = batch.ctrl if use_ctrl else torch.zeros_like(batch.ctrl)
        pred = model(x_masked, mask, ctrl, batch.edge_index)
        masked_idx = mask.bool()
        real_all.append(batch.x[masked_idx].cpu())
        pred_all.append(pred[masked_idx].cpu())
    real = torch.cat(real_all).numpy()
    pred = torch.cat(pred_all).numpy()

    per_feature_corr, per_feature_mse = {}, {}
    for i, name in enumerate(FEATURE_NAMES):
        mse = float(np.mean((real[:, i] - pred[:, i]) ** 2))
        corr = float(np.corrcoef(real[:, i], pred[:, i])[0, 1]) if np.std(pred[:, i]) > 1e-8 else float("nan")
        per_feature_mse[name] = mse
        per_feature_corr[name] = corr
    return per_feature_mse, per_feature_corr, real, pred


def main(steps: int = 2000, tag: str = "norm", mask_frac: float = 0.3):
    dev = device()
    print("loading graphs + extracting control fields ...")
    graphs, ctrls = load_graphs_with_ctrl(tag)
    all_x = torch.cat([g.x for g in graphs], dim=0)
    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    for g in graphs:
        g.x = (g.x - mean) / std
    print(f"loaded {len(graphs)} graphs, {sum(g.num_nodes for g in graphs)} total nodes")

    print("\ntraining WITH control-field conditioning ...")
    model_ctrl, hist_ctrl = train_one(True, graphs, ctrls, mean, std, steps, 16, mask_frac, dev, seed=0)
    print("\ntraining WITHOUT control-field conditioning (ablation) ...")
    model_noctrl, hist_noctrl = train_one(False, graphs, ctrls, mean, std, steps, 16, mask_frac, dev, seed=0)

    print("\nevaluating both (same held-out-style masked subgraphs, seed=1) ...")
    mse_ctrl, corr_ctrl, real_c, pred_c = evaluate(model_ctrl, True, graphs, ctrls, dev, seed=1)
    mse_noctrl, corr_noctrl, real_n, pred_n = evaluate(model_noctrl, False, graphs, ctrls, dev, seed=1)

    print(f"\n{'feature':<28} {'corr w/ ctrl':>14} {'corr no ctrl':>14} {'mse w/ ctrl':>14} {'mse no ctrl':>14}")
    for name in FEATURE_NAMES:
        print(f"{name:<28} {corr_ctrl[name]:>14.3f} {corr_noctrl[name]:>14.3f} "
              f"{mse_ctrl[name]:>14.3f} {mse_noctrl[name]:>14.3f}")

    overall_mse_ctrl = float(np.mean(list(mse_ctrl.values())))
    overall_mse_noctrl = float(np.mean(list(mse_noctrl.values())))
    mean_corr_ctrl = float(np.nanmean(list(corr_ctrl.values())))
    mean_corr_noctrl = float(np.nanmean(list(corr_noctrl.values())))
    print(f"\noverall: mean MSE with_ctrl={overall_mse_ctrl:.4f} no_ctrl={overall_mse_noctrl:.4f}")
    print(f"overall: mean corr with_ctrl={mean_corr_ctrl:.4f} no_ctrl={mean_corr_noctrl:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model_ctrl.state_dict(), OUT_DIR / "infill_model_ctrl.pt")
    torch.save(model_noctrl.state_dict(), OUT_DIR / "infill_model_noctrl.pt")
    np.savez(
        OUT_DIR / "infill_eval.npz",
        feature_names=np.array(FEATURE_NAMES),
        mse_ctrl=np.array([mse_ctrl[n] for n in FEATURE_NAMES]),
        mse_noctrl=np.array([mse_noctrl[n] for n in FEATURE_NAMES]),
        corr_ctrl=np.array([corr_ctrl[n] for n in FEATURE_NAMES]),
        corr_noctrl=np.array([corr_noctrl[n] for n in FEATURE_NAMES]),
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(FEATURE_NAMES))
    w = 0.35
    ax.bar(x_pos - w / 2, [corr_ctrl[n] for n in FEATURE_NAMES], w, label="with control fields")
    ax.bar(x_pos + w / 2, [corr_noctrl[n] for n in FEATURE_NAMES], w, label="without (ablation)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURE_NAMES, rotation=60, ha="right")
    ax.set_ylabel("corr(real, predicted) on masked nodes")
    ax.set_title(f"R2 infill: control-field conditioning ablation (mask_frac={mask_frac}, steps={steps})")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    out_path = OUT_DIR / "infill_ablation.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(steps=steps)
