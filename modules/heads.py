from __future__ import annotations
import torch
from torch import nn


class GaussianAffinityHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)

    def forward(self, ab_vec: torch.Tensor, ag_vec: torch.Tensor) -> dict:
        fused = torch.cat([ab_vec, ag_vec], dim=-1)
        features = self.trunk(fused)
        mu = self.mu(features).squeeze(-1)
        log_var = self.log_var(features).squeeze(-1)
        return {"mu": mu, "log_var": log_var}
