"""R2 minimal viable graph generator, scoped down per ROADMAP2: a node-
attribute INFILL model, not full autoregressive growth+attachment
generation. Given a subgraph with some nodes' attributes masked but its real
topology known, plus per-node control-field conditioning (size_gradient
target, local alignment flow), predict the masked nodes' true attribute
vectors. Deliberately does not reuse GraphVAE.decode() -- R1 found that
decoder reconstructs position well but not intrinsic shape/size attributes,
so this is a fresh, purpose-built model for the attribute-prediction task.

Topology/attachment prediction (which existing cell(s) a new cell connects
to) is out of scope for this first pass -- this only tests whether control
fields improve attribute infill given already-known adjacency, the smaller
half of the R2 problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class NodeInfillGNN(nn.Module):
    def __init__(self, attr_dim: int, ctrl_dim: int, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        # Input per node: masked attrs (zeroed where masked) + mask flag (1) + control fields.
        in_dim = attr_dim + 1 + ctrl_dim
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.out = nn.Linear(hidden_dim, attr_dim)

    def forward(self, x_masked, mask_flag, ctrl, edge_index):
        h = torch.cat([x_masked, mask_flag.unsqueeze(-1), ctrl], dim=-1)
        h = F.elu(self.conv1(h, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        h = F.elu(self.conv3(h, edge_index))
        return self.out(h)
