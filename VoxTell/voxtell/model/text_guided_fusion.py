import torch
from torch import nn


class TokenToVoxelCrossAttention(nn.Module):
    """
    CRIS-lite prompt-specific dense text fusion.

    Voxel features attend to token embeddings so the image memory is conditioned
    by fine-grained text tokens before the global query decoder runs.
    """

    def __init__(
        self,
        voxel_dim: int,
        text_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.base_residual_scale = residual_scale
        self.runtime_scale = 1.0
        self.last_delta_mean: float | None = None
        self.last_gate_mean: float | None = None
        self.voxel_norm = nn.LayerNorm(voxel_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.query_proj = nn.Linear(voxel_dim, hidden_dim)
        self.key_proj = nn.Linear(text_dim, hidden_dim)
        self.value_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, voxel_dim),
            nn.GELU(),
            nn.Linear(voxel_dim, voxel_dim),
        )
        self.prompt_gate = nn.Sequential(
            nn.Linear(text_dim, voxel_dim),
            nn.Sigmoid(),
        )

    def set_strength(self, strength: float) -> None:
        self.runtime_scale = max(0.0, float(strength))

    def forward(
        self,
        voxel_feat: torch.Tensor,
        pooled_text: torch.Tensor,
        token_text: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_voxel = self.voxel_norm(voxel_feat)
        norm_text = self.text_norm(token_text)
        query = self.query_proj(norm_voxel)
        key = self.key_proj(norm_text)
        value = self.value_proj(norm_text)
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.bool()
        ctx, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        residual = self.out_proj(ctx)
        gate = self.prompt_gate(pooled_text).unsqueeze(1)
        delta = self.base_residual_scale * self.runtime_scale * gate * residual
        self.last_delta_mean = float(delta.detach().abs().mean().cpu())
        self.last_gate_mean = float(gate.detach().mean().cpu())
        return voxel_feat + delta


class ScaleAwarePromptModulation(nn.Module):
    """
    STPNet-lite scale-aware prompt refinement.

    A pooled prompt context generates different residuals for different decoder
    scales so fine-detail and coarse localization can react differently.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_scales: int,
        hidden_dim: int = 1024,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.base_residual_scale = residual_scale
        self.runtime_scale = 1.0
        self.context_norm = nn.LayerNorm(context_dim)
        self.prompt_norm = nn.LayerNorm(query_dim)
        self.to_scale_context = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales * query_dim),
        )
        self.to_scale_gate = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_scales * query_dim),
            nn.Sigmoid(),
        )

    def set_strength(self, strength: float) -> None:
        self.runtime_scale = max(0.0, float(strength))

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        norm_prompt = self.prompt_norm(prompt_embedding)
        norm_context = self.context_norm(context_embedding)
        scale_context = self.to_scale_context(norm_context).view(
            norm_context.shape[0], self.num_scales, norm_prompt.shape[-1]
        )
        scale_gate = self.to_scale_gate(norm_context).view(
            norm_context.shape[0], self.num_scales, norm_prompt.shape[-1]
        )
        outputs: list[torch.Tensor] = []
        for scale_idx in range(self.num_scales):
            residual = scale_context[:, scale_idx:scale_idx + 1, :]
            gate = scale_gate[:, scale_idx:scale_idx + 1, :]
            outputs.append(
                prompt_embedding
                + self.base_residual_scale * self.runtime_scale * gate * residual
                + 0.05 * norm_prompt
            )
        return outputs
