import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossZG(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, g):
        z_norm = F.normalize(z, dim=1)
        g_norm = F.normalize(g, dim=1)
        sim = torch.mm(z_norm, g_norm.t()) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(sim)
        return -torch.log(torch.diag(exp) / exp.sum(dim=1)).mean()


class ContrastiveLossZL(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, l):
        z_norm = F.normalize(z, dim=1)
        l_norm = F.normalize(l, dim=1)
        sim = torch.mm(z_norm, l_norm.t()) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(sim)
        return -torch.log(torch.diag(exp) / exp.sum(dim=1)).mean()
