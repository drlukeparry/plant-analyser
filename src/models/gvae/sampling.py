"""Subgraph sampling: turns one whole-image cell graph into many training
examples (k-hop neighborhoods around random seed cells), per Phase 3 plan —
gives thousands of overlapping local "tissue patches" from a handful of images.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def sample_subgraph(data: Data, num_hops: int = 3, max_nodes: int = 400) -> Data:
    seed = torch.randint(0, data.num_nodes, (1,)).item()
    node_idx, edge_index, _, edge_mask = k_hop_subgraph(
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
        edge_mask_local = edge_sel
    else:
        edge_mask_local = torch.ones(edge_index.size(1), dtype=torch.bool)

    x = data.x[node_idx]
    edge_attr = data.edge_attr[edge_mask][edge_mask_local] if data.edge_attr is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SubgraphSampler:
    """Infinite iterator of random subgraphs from one or more source graphs."""

    def __init__(self, graphs: list[Data], num_hops: int = 3, max_nodes: int = 400):
        self.graphs = graphs
        self.num_hops = num_hops
        self.max_nodes = max_nodes

    def sample(self) -> Data:
        g = self.graphs[torch.randint(0, len(self.graphs), (1,)).item()]
        return sample_subgraph(g, self.num_hops, self.max_nodes)

    def sample_batch(self, batch_size: int):
        from torch_geometric.data import Batch

        subs = [self.sample() for _ in range(batch_size)]
        subs = [s for s in subs if s.num_nodes >= 8]  # drop degenerate tiny subgraphs
        return Batch.from_data_list(subs)
