"""Graph-VAE: GAT encoder -> attention pooling -> VAE bottleneck -> GNN decoder
reconstructing node features, + inner-product decoder for adjacency.
Per PLAN.md Phase 3: latent z is the per-subgraph "tissue composition" descriptor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation


class GraphVAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 32, heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: node features + local topology -> pooled graph embedding -> (mu, logvar)
        self.enc_conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.enc_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z broadcast to every node as identical initial features, then
        # message-passing over the SAME input topology differentiates nodes by
        # their structural position (different neighborhoods -> different
        # embeddings, even though every node starts from the same z).
        self.dec_conv1 = GATConv(latent_dim, hidden_dim, heads=heads, concat=True)
        self.dec_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.node_out = nn.Linear(hidden_dim, in_dim)
        self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x, edge_index, batch):
        h = F.elu(self.enc_conv1(x, edge_index))
        h = F.elu(self.enc_conv2(h, edge_index))
        pooled = self.pool(h, batch)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, edge_index, batch):
        z_per_node = z[batch]
        h = F.elu(self.dec_conv1(z_per_node, edge_index))
        h = F.elu(self.dec_conv2(h, edge_index))
        node_recon = self.node_out(h)
        edge_embed = self.edge_embed(h)
        return node_recon, edge_embed

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        node_recon, edge_embed = self.decode(z, edge_index, batch)
        return node_recon, edge_embed, mu, logvar
