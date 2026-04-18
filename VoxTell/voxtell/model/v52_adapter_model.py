from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

from voxtell.model.chest_text_guided_adapter import ChestTextGuidedAdapter
from voxtell.model.voxtell_model import VoxTellModel


class VoxTellV52AdapterModel(nn.Module):
    """
    Conservative v5.2 adapter wrapper:
    - keeps the original VoxTell decoder/query path
    - applies a lightweight text-guided adapter at pre-decoder or post-decoder
    - disables category bias by default
    - exposes suppression tensors for explicit supervision
    """

    def __init__(
        self,
        base_model: VoxTellModel,
        adapter_hidden_dim: int,
        adapter_insertion_point: str,
        adapter_num_groups: int,
        adapter_suppression_groups: int | None,
        adapter_residual_scale: float,
        adapter_gate_cap: float,
        use_category_bias: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter_insertion_point = adapter_insertion_point
        self.text_guided_adapter = ChestTextGuidedAdapter(
            feature_dim=base_model.query_dim,
            text_dim=base_model.text_embedding_dim,
            hidden_dim=adapter_hidden_dim,
            num_groups=adapter_num_groups,
            suppression_groups=adapter_suppression_groups,
            residual_scale=adapter_residual_scale,
            gate_cap=adapter_gate_cap,
            use_category_bias=use_category_bias,
        )
        self.last_suppression_mean: float | None = None
        self.last_suppression_tensor: torch.Tensor | None = None
        self.last_token_fusion_delta_mean: float | None = None
        self.last_token_gate_mean: float | None = None
        self.last_query_scale: float | None = 1.0

    @property
    def query_dim(self) -> int:
        return self.base_model.query_dim

    def set_fusion_schedule(self, fusion_strength: float, scale_strength: float | None = None) -> None:
        # Kept for training-script compatibility; v5.2 does not use token fusion or scale prompting.
        return

    def set_query_schedule(self, query_strength: float) -> None:
        # Kept for training-script compatibility; v5.2 keeps the original query path unchanged.
        self.last_query_scale = 1.0

    def forward(
        self,
        img: torch.Tensor,
        text_embedding: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        text_token_embeddings: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        category_text_embedding: torch.Tensor | None = None,
        category_token_embeddings: torch.Tensor | None = None,
        category_attention_mask: torch.Tensor | None = None,
    ):
        del text_token_embeddings, text_attention_mask, category_text_embedding, category_token_embeddings, category_attention_mask

        skips = self.base_model.encoder(img)
        selected_feature = skips[self.base_model.selected_decoder_layer]
        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = self.base_model.project_bottleneck_embed(bottleneck_embed)
        bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> b (h w d) c")

        text_embedding = text_embedding.squeeze(2)
        num_prompts = text_embedding.shape[1]
        text_query_embed = self.base_model.project_text_embed(text_embedding)

        self.last_token_fusion_delta_mean = None
        self.last_token_gate_mean = None
        self.last_query_scale = 1.0

        outs = []
        for prompt_idx in range(num_prompts):
            prompt_text = text_embedding[:, prompt_idx : prompt_idx + 1]
            query_embed = text_query_embed[:, prompt_idx : prompt_idx + 1]
            prompt_category_ids = category_ids[:, prompt_idx : prompt_idx + 1] if category_ids is not None else None

            prompt_memory = bottleneck_embed
            if self.adapter_insertion_point == "pre_decoder":
                prompt_memory = self.text_guided_adapter(
                    prompt_memory,
                    prompt_text,
                    category_ids=prompt_category_ids,
                )

            self.last_suppression_mean = self.text_guided_adapter.last_suppression_mean
            self.last_suppression_tensor = self.text_guided_adapter.last_suppression_tensor

            memory = rearrange(prompt_memory, "b m c -> m b c")
            query = repeat(query_embed, "b n dim -> n b dim")
            mask_embedding, _ = self.base_model.transformer_decoder(
                tgt=query,
                memory=memory,
                pos=self.base_model.pos_embed,
                memory_key_padding_mask=None,
            )
            mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")

            if self.adapter_insertion_point == "post_decoder":
                mask_embedding = self.text_guided_adapter(
                    mask_embedding,
                    prompt_text,
                    category_ids=prompt_category_ids,
                )
                self.last_suppression_mean = self.text_guided_adapter.last_suppression_mean
                self.last_suppression_tensor = self.text_guided_adapter.last_suppression_tensor

            prompt_embeds = [
                projection(mask_embedding)
                for projection in self.base_model.project_to_decoder_channels
            ]
            outs.append(self.base_model.decoder(skips, prompt_embeds))

        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        if not self.base_model.deep_supervision:
            outs = outs[0]
        return outs
