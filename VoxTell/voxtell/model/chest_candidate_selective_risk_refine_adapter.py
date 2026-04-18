from typing import Optional

import torch
from torch import nn


class ChestCandidateSelectiveRiskRefineAdapter(nn.Module):
    """
    v6.7 adapter:
    - keeps candidate/risk/refine decomposition from v6.5
    - turns risk into bounded selective suppression inside candidate-support regions
    - keeps refine as a weak uncertainty-guided boundary cleanup path
    """

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
        risk_groups: int | None = None,
        candidate_scale: float = 0.1,
        risk_scale: float = 0.08,
        gate_cap: float = 0.25,
        num_categories: int = 16,
        use_category_bias: bool = True,
        category_scale: float = 0.2,
        refine_scale: float = 0.035,
    ) -> None:
        super().__init__()
        self.num_groups = max(1, min(num_groups, feature_dim))
        self.risk_groups = max(1, min(risk_groups or self.num_groups, feature_dim))
        self.candidate_scale = candidate_scale
        self.risk_scale = risk_scale
        self.gate_cap = gate_cap
        self.use_category_bias = use_category_bias
        self.category_scale = category_scale
        self.refine_scale = refine_scale

        self.text_norm = nn.LayerNorm(text_dim)
        self.category_embedding = nn.Embedding(num_categories, text_dim) if use_category_bias else None

        self.to_logit_context = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),
        )
        self.logit_candidate_head = nn.Sequential(
            nn.Conv3d(6, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1),
        )
        self.logit_risk_head = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1),
        )
        self.logit_refine_head = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1),
        )

        self.last_candidate_mean: float | None = None
        self.last_risk_mean: float | None = None
        self.last_risk_tensor: torch.Tensor | None = None
        self.last_refine_mean: float | None = None
        self.last_refine_tensor: torch.Tensor | None = None

    def forward(
        self,
        feature: torch.Tensor,
        text_embedding: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del text_embedding, category_ids
        self.last_candidate_mean = None
        self.last_risk_mean = None
        self.last_risk_tensor = None
        self.last_refine_mean = None
        self.last_refine_tensor = None
        return feature

    def rectify_logits(
        self,
        base_logits: torch.Tensor,
        text_embedding: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        combined_text = self._combine_text(text_embedding, category_ids)
        if combined_text.ndim == 3:
            combined_text = combined_text.squeeze(1)

        context = self.to_logit_context(combined_text).to(base_logits.dtype)
        context = context.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        context = context.expand(-1, -1, *base_logits.shape[-3:])

        base_prob = torch.sigmoid(base_logits)
        candidate_input = torch.cat([base_logits, base_prob, context], dim=1)
        candidate_delta = self.logit_candidate_head(candidate_input)
        candidate_logits = base_logits + self.candidate_scale * candidate_delta
        candidate_prob = torch.sigmoid(candidate_logits)
        candidate_uncertainty = 4.0 * candidate_prob * (1.0 - candidate_prob)
        binary_hint = (candidate_prob > 0.5).float()

        risk_input = torch.cat([candidate_logits, candidate_prob, candidate_uncertainty, binary_hint, context], dim=1)
        risk_logit = self.logit_risk_head(risk_input)
        risk_prob = torch.sigmoid(risk_logit)
        effective_risk = risk_prob * torch.sqrt(candidate_prob.detach().clamp_min(1e-6))
        risk_bias = self.risk_scale * effective_risk

        refine_logit = self.logit_refine_head(risk_input)
        refine_prob = torch.sigmoid(refine_logit)
        effective_refine = refine_prob * candidate_uncertainty.detach()
        refine_bias = self.refine_scale * effective_refine

        self.last_candidate_mean = float(candidate_prob.detach().mean().cpu())
        self.last_risk_mean = float(effective_risk.detach().mean().cpu())
        self.last_risk_tensor = effective_risk
        self.last_refine_mean = float(effective_refine.detach().mean().cpu())
        self.last_refine_tensor = effective_refine
        return candidate_logits - risk_bias - refine_bias

    def _combine_text(
        self,
        text_embedding: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_text = self.text_norm(text_embedding)
        if category_ids is not None and self.category_embedding is not None:
            if category_ids.ndim == 1:
                category_ids = category_ids.unsqueeze(1)
            cat_embed = self.category_embedding(category_ids)
            return norm_text + self.category_scale * cat_embed
        return norm_text
