import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


def make_conv_layer(hidden_dim, num_heads, dropout):
    return GATConv(in_channels=hidden_dim, out_channels=hidden_dim // num_heads,
                   heads=num_heads, concat=True, edge_dim=hidden_dim,
                   dropout=dropout, add_self_loops=False)


def _conv_forward(conv, h, edge_index, edge_emb):
    return conv(h, edge_index, edge_attr=edge_emb)


class MultiHeadGlobalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, batch):

        batch_size = batch.max().item() + 1

        Q = self.query(x).view(-1, self.num_heads, self.head_dim)
        K = self.key(x).view(-1, self.num_heads, self.head_dim)
        V = self.value(x).view(-1, self.num_heads, self.head_dim)

        out_list = []
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() == 0:
                continue

            q = Q[mask]
            k = K[mask]
            v = V[mask]

            scores = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)

            output = torch.einsum('nhm,mhd->nhd', attn, v)
            out_list.append(output)

        out = torch.cat(out_list, dim=0).reshape(-1, self.num_heads * self.head_dim)
        out = self.out(out)

        return out


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim=44,
        edge_feat_dim=10,
        lap_pe_dim=8,
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        use_gatv2=True,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.pe_encoder = nn.Linear(lap_pe_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(make_conv_layer(hidden_dim, num_heads, dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.residual_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.global_attention = MultiHeadGlobalAttention(hidden_dim, num_heads=num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, lap_pe, batch):
        h = self.node_encoder(x)

        h_pe = self.pe_encoder(lap_pe)
        h = h + h_pe

        edge_emb = self.edge_encoder(edge_attr)

        intermediate_pools = []

        for i in range(self.num_layers):
            h_res = self.residual_layers[i](h)

            h_new = _conv_forward(self.convs[i], h, edge_index, edge_emb)
            h_new = self.batch_norms[i](h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)

            h = h_res + h_new

            pooled = global_mean_pool(h, batch)
            intermediate_pools.append(pooled)

        h = self.global_attention(h, batch)
        graph_emb_mean = global_mean_pool(h, batch)
        graph_emb_max  = global_max_pool(h, batch)

        graph_embedding = torch.cat([graph_emb_mean, graph_emb_max], dim=-1)
        return graph_embedding, intermediate_pools
