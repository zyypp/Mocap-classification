import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization: A_hat = D^{-1/2} (A + I) D^{-1/2}
    Accepts [N,N] or [B,N,N]; returns same shape.
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    I = torch.eye(A.size(1), device=A.device).unsqueeze(0).expand_as(A)
    A_tilde = A + I
    D = A_tilde.sum(dim=-1)  # [B,N]
    D_inv_sqrt = torch.pow(D.clamp(min=1e-8), -0.5)
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    if squeeze:
        A_hat = A_hat.squeeze(0)
    return A_hat


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, F]
        A_hat: [N, N] or [B, N, N]
        returns: [B, N, out]
        """
        h = self.lin(x)
        if A_hat.dim() == 2:
            return torch.matmul(A_hat, h)  # broadcast [N,N] x [B,N,out] -> [B,N,out]
        return torch.bmm(A_hat, h)


class GCNClassifier(nn.Module):
    def __init__(self, num_nodes: int, in_node_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        assert num_layers >= 1
        self.num_nodes = num_nodes
        self.layers = nn.ModuleList()
        dims = [in_node_dim] + [hidden_dim] * (num_layers - 1)
        for d_in in dims:
            self.layers.append(GraphConv(d_in, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_nodes: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x_nodes: [B, N, in_node_dim] where N=num_nodes
        A: [N, N] or [B, N, N]
        returns logits: [B, num_classes]
        """
        A_hat = normalize_adjacency(A)
        h = x_nodes
        for gc in self.layers:
            h = gc(h, A_hat)
            h = F.relu(h)
            h = self.dropout(h)
        # Global mean pool over nodes
        g = h.mean(dim=1)  # [B, hidden]
        logits = self.classifier(g)
        return logits


