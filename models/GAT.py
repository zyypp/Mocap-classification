import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GATConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1, alpha: float = 0.2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.out_per_head))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, F]
        adj: [N, N] (fixed) or [B, N, N]
        returns: [B, N, out_features]
        """
        B, N, _ = x.shape
        h = self.lin(x)  # [B,N,out]
        h = h.view(B, N, self.num_heads, self.out_per_head)  # [B,N,H,D]
        # Compute attention scores e_ij per head
        src = (h * self.a_src)  # [B,N,H,D]
        dst = (h * self.a_dst)  # [B,N,H,D]
        e_src = src.sum(dim=-1)  # [B,N,H]
        e_dst = dst.sum(dim=-1)  # [B,N,H]
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1)  # [B,i,j,H]
        e = self.leaky_relu(e)
        # Mask with adjacency (where adj==0, set to -inf so softmax->0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        # add self-loops to ensure at least one valid neighbor per node
        I = torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype).unsqueeze(0).expand_as(adj)
        adj = (adj > 0) | (I > 0)
        mask = adj.unsqueeze(-1)  # [B,N,N,1] boolean
        e = e.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(e, dim=2)  # softmax over neighbors j
        attn = self.dropout(attn)
        # Aggregate from neighbors j to target i: attn[b,i,j,h] * h[b,j,h,d] -> out[b,i,h,d]
        h_out = torch.einsum('bijh,bjhd->bihd', attn, h)  # [B,N,H,D]
        h_out = h_out.reshape(B, N, self.num_heads * self.out_per_head)
        return h_out


class GATClassifier(nn.Module):
    def __init__(self, num_nodes: int, in_node_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        d_in = in_node_dim
        for _ in range(num_layers):
            self.layers.append(GATConv(d_in, hidden_dim, num_heads=num_heads, dropout=dropout))
            d_in = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_nodes: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = x_nodes
        for gat in self.layers:
            h = gat(h, A)
            h = F.elu(h)
            h = self.dropout(h)
        g = h.mean(dim=1)
        logits = self.classifier(g)
        return logits


