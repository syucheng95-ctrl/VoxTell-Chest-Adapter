import torch
from torch import nn


class ChestTextGuidedAdapter(nn.Module):
    """
    Lightweight FiLM-style adapter that uses text embeddings to modulate
    prompt-specific latent features before they are projected into decoder space.

    Input:
        feature: (B, N, C)
        text_embedding: (B, N, D)
    Output:
        adapted feature: (B, N, C)
    """

    def __init__(
        self,
        feature_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_groups: int = 8,
        residual_scale: float = 0.1,
        gate_cap: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_groups = max(1, min(num_groups, feature_dim))
        self.residual_scale = residual_scale
        self.gate_cap = gate_cap
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.to_delta = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.to_group_gate = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.num_groups),
        )

    def forward(self, feature: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        norm_feature = self.feature_norm(feature)
        norm_text = self.text_norm(text_embedding)
        delta = torch.tanh(self.to_delta(norm_text))
        group_gate = torch.sigmoid(self.to_group_gate(norm_text)) * self.gate_cap
        channel_gate = self._expand_group_gate(group_gate, feature.shape[-1])
        return feature + self.residual_scale * channel_gate * delta * norm_feature

    def _expand_group_gate(self, gate: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if gate.shape[-1] == feature_dim:
            return gate
        base = feature_dim // gate.shape[-1]
        remainder = feature_dim % gate.shape[-1]
        repeats = [base + (1 if idx < remainder else 0) for idx in range(gate.shape[-1])]
        expanded = [gate[..., idx:idx + 1].expand(*gate.shape[:-1], repeats[idx]) for idx in range(gate.shape[-1])]
        return torch.cat(expanded, dim=-1)
