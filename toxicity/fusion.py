import torch
import torch.nn as nn


class BCMGFusion(nn.Module):

    def __init__(self, d_g, d_l, d_out, rank=128, num_heads=4):
        super().__init__()
        self.U       = nn.Linear(d_g, rank)
        self.V       = nn.Linear(d_l, rank)
        self.gate_gl = nn.Linear(rank, rank)
        self.gate_lg = nn.Linear(rank, rank)
        self.W       = nn.Linear(2 * rank, d_out)

    def forward(self, d_g: torch.Tensor, d_l: torch.Tensor):
        u       = self.U(d_g)
        v       = self.V(d_l)
        u_gated = u * torch.sigmoid(self.gate_gl(v))
        v_gated = v * torch.sigmoid(self.gate_lg(u))
        fused   = torch.cat([u_gated, v_gated], dim=-1)
        return self.W(fused), None
