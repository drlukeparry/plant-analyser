"""D1: fixed-cardinality tokenized dataset for the set/graph DDIM.

Reuses ROADMAP2's control-field extraction (control_fields.py, already
sanity-checked -- see ROADMAP2.md R2) and Phase 3's k-hop subgraph sampler
unchanged. Adjacency (edge_index) is deliberately dropped from what's
returned to the model -- per ROADMAP3.md, edges are not part of the
generative target, only the k-hop sampling is reused to draw spatially
local, plausible-sized cell neighborhoods to train on.

Padding to a fixed N_max is a training/batching convenience only, not an
architectural requirement -- the Transformer denoiser works over any
sequence length, so sampling (roadmap3/sample.py) uses each real subgraph's
actual node count instead of padding.
"""
import torch
from torch_geometric.utils import k_hop_subgraph

from src.roadmap2.train_infill import load_graphs_with_ctrl  # noqa: F401 (re-exported for convenience)


def sample_token_subgraph(data, ctrl, num_hops: int = 3, max_nodes: int = 300):
    """One real, spatially-local set of cells: (x, ctrl), no edges."""
    seed = torch.randint(0, data.num_nodes, (1,)).item()
    node_idx, _, _, _ = k_hop_subgraph(
        seed, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    if node_idx.numel() > max_nodes:
        keep = torch.randperm(node_idx.numel())[:max_nodes]
        node_idx = node_idx[keep]
    return data.x[node_idx], ctrl[node_idx]


class FixedSetSampler:
    """Draws batches of shape [B, n_max, attr_dim] / [B, n_max, ctrl_dim] with
    an [B, n_max] active mask, padding real (variable-size) token subgraphs
    with zeros. n_max only needs to be >= the largest subgraph actually
    drawn; pick it from the real subgraph size distribution (num_hops=3,
    max_nodes=300 gives subgraphs well under 300 in practice for this
    dataset's typical local cell density -- verify empirically before
    committing to a specific n_max for a training run)."""

    def __init__(self, graphs, ctrls, n_max: int, num_hops: int = 3, max_nodes: int = 300):
        self.graphs, self.ctrls = graphs, ctrls
        self.n_max, self.num_hops, self.max_nodes = n_max, num_hops, max_nodes

    def _sample_one(self):
        i = torch.randint(0, len(self.graphs), (1,)).item()
        x, c = sample_token_subgraph(self.graphs[i], self.ctrls[i], self.num_hops, self.max_nodes)
        n = min(x.size(0), self.n_max)
        x, c = x[:n], c[:n]
        return x, c, n

    def sample_batch(self, batch_size: int):
        attr_dim = self.graphs[0].x.size(1)
        ctrl_dim = self.ctrls[0].size(1)
        x_out = torch.zeros(batch_size, self.n_max, attr_dim)
        c_out = torch.zeros(batch_size, self.n_max, ctrl_dim)
        active = torch.zeros(batch_size, self.n_max)
        for b in range(batch_size):
            x, c, n = self._sample_one()
            if n < 8:  # too small a neighborhood to be a meaningful training example
                x, c, n = self._resample_until_large(min_n=8)
            x_out[b, :n], c_out[b, :n], active[b, :n] = x, c, 1.0
        return x_out, c_out, active

    def _resample_until_large(self, min_n: int, max_tries: int = 10):
        for _ in range(max_tries):
            x, c, n = self._sample_one()
            if n >= min_n:
                return x, c, n
        return x, c, n  # give up gracefully after max_tries, still usable
