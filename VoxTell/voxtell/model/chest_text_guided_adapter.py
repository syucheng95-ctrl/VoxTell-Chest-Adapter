from typing import Optional

import torch
from torch import nn


class ChestTextGuidedAdapter(nn.Module):
    """
    ChestTextGuidedAdapter v4-ready:
    - Category-aware gating: Uses category embeddings to refine text modulation.
    - Group-wise suppression branch: Explicitly dampens non-relevant feature groups.

    Input:
        feature: (B, N, C)
        text_embedding: (B, N, D)
        category_ids: Optional (B,) tensor of category indices
    Output:
        adapted feature: (B, N, C)
    """

    # ReXGroundingCT category codes. Keep the raw dataset codes instead of
    # collapsing them into broad lesion labels so the adapter can learn
    # task-specific modulation directly from the benchmark taxonomy.
    CATEGORY_MAP = {
        "none": 0,
        "1a": 1,
        "1b": 2,
        "1c": 3,
        "1d": 4,
        "1e": 5,
        "1f": 6,
        "2a": 7,
        "2b": 8,
        "2c": 9,
        "2d": 10,
        "2e": 11,
        "2f": 12,
        "2g": 13,
        "2h": 14,
        "other": 15,
    }

    def __init__(
        self,
        feature_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_groups: int = 8,
        suppression_groups: int | None = None,
        residual_scale: float = 0.1,
        gate_cap: float = 0.25,
        num_categories: int = 16,
        suppression_strength: float = 0.2,
        use_category_bias: bool = True,
        category_scale: float = 1.0,
        suppression_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_groups = max(1, min(num_groups, feature_dim))
        self.suppression_groups = max(1, min(suppression_groups or self.num_groups, feature_dim))
        self.residual_scale = residual_scale
        self.gate_cap = gate_cap
        self.suppression_strength = suppression_strength
        self.use_category_bias = use_category_bias
        self.category_scale = category_scale
        self.suppression_scale = suppression_scale

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.text_norm = nn.LayerNorm(text_dim)

        # Category embedding to provide specific bias for different findings
        self.category_embedding = nn.Embedding(num_categories, text_dim) if use_category_bias else None

        # Modulation branch (v2 base)
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

        # Group-wise suppression branch.
        self.to_suppression = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.suppression_groups),
            nn.Sigmoid()
        )
        self.last_suppression_mean: float | None = None
        self.last_suppression_tensor: torch.Tensor | None = None

    def forward(
        self,
        feature: torch.Tensor,
        text_embedding: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Normalize and integrate category info
        norm_feature = self.feature_norm(feature)
        norm_text = self.text_norm(text_embedding)

        if category_ids is not None and self.category_embedding is not None:
            if category_ids.ndim == 1:
                category_ids = category_ids.unsqueeze(1)
            # category embedding now matches (B, N, D)
            cat_embed = self.category_embedding(category_ids)
            combined_text = norm_text + self.category_scale * cat_embed
        else:
            combined_text = norm_text

        # 2. Compute modulation (Delta & Gating)
        delta = torch.tanh(self.to_delta(combined_text))
        group_gate = torch.sigmoid(self.to_group_gate(combined_text)) * self.gate_cap
        channel_gate = self._expand_group_gate(group_gate, feature.shape[-1])

        # 3. Compute group-wise suppression signal
        suppression = self.to_suppression(combined_text)  # (B, N, G)
        suppression_gate = self._expand_group_gate(suppression, feature.shape[-1])
        self.last_suppression_mean = float(suppression.detach().mean().cpu())
        self.last_suppression_tensor = suppression
        
        # 4. Final fusion: keep enhancement and suppression in the same group space.
        modulated_residual = self.residual_scale * channel_gate * delta * norm_feature
        suppression_factor = 1.0 - (self.suppression_strength * self.suppression_scale) * (
            1.0 - suppression_gate
        )
        final_feature = feature + modulated_residual * suppression_factor

        return final_feature

    def _expand_group_gate(self, gate: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if gate.shape[-1] == feature_dim:
            return gate
        base = feature_dim // gate.shape[-1]
        remainder = feature_dim % gate.shape[-1]
        repeats = [base + (1 if idx < remainder else 0) for idx in range(gate.shape[-1])]
        expanded = [gate[..., idx:idx + 1].expand(*gate.shape[:-1], repeats[idx]) for idx in range(gate.shape[-1])]
        return torch.cat(expanded, dim=-1)
