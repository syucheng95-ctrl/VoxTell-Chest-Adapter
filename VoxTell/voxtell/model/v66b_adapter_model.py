from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn

from voxtell.model.chest_masked_risk_refine_adapter_v66b import ChestMaskedRiskRefineAdapterV66b
from voxtell.model.voxtell_model import VoxTellModel


class VoxTellV66bAdapterModel(nn.Module):
    """
    v6.6b wrapper:
    - keeps the v6.5/v6.6 candidate-risk-refine structure
    - uses bounded masked risk
    - uses uncertainty-guided refine without extra gate hyperparameters
    """

    def __init__(
        self,
        base_model: VoxTellModel,
        adapter_hidden_dim: int,
        adapter_insertion_point: str,
        adapter_num_groups: int,
        adapter_risk_groups: int | None,
        adapter_candidate_scale: float,
        adapter_risk_scale: float,
        adapter_gate_cap: float,
        adapter_category_scale: float,
        adapter_refine_scale: float,
        use_category_bias: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter_insertion_point = adapter_insertion_point
        self.text_guided_adapter = ChestMaskedRiskRefineAdapterV66b(
            feature_dim=base_model.query_dim,
            text_dim=base_model.text_embedding_dim,
            hidden_dim=adapter_hidden_dim,
            num_groups=adapter_num_groups,
            risk_groups=adapter_risk_groups,
            candidate_scale=adapter_candidate_scale,
            risk_scale=adapter_risk_scale,
            gate_cap=adapter_gate_cap,
            use_category_bias=use_category_bias,
            category_scale=adapter_category_scale,
            refine_scale=adapter_refine_scale,
        )
        self.last_candidate_mean: float | None = None
        self.last_risk_mean: float | None = None
        self.last_risk_tensor: torch.Tensor | None = None
        self.last_suppression_mean: float | None = None
        self.last_suppression_tensor: torch.Tensor | None = None
        self.last_suppression_raw_mean: float | None = None
        self.last_suppression_raw_tensor: torch.Tensor | None = None
        self.last_token_fusion_delta_mean: float | None = None
        self.last_token_gate_mean: float | None = None
        self.last_query_scale: float | None = 1.0

    @property
    def query_dim(self) -> int:
        return self.base_model.query_dim

    def set_fusion_schedule(self, fusion_strength: float, scale_strength: float | None = None) -> None:
        return

    def set_query_schedule(self, query_strength: float) -> None:
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

            self.last_candidate_mean = self.text_guided_adapter.last_candidate_mean
            self.last_risk_mean = self.text_guided_adapter.last_risk_mean
            self.last_risk_tensor = self.text_guided_adapter.last_risk_tensor
            self.last_suppression_mean = self.text_guided_adapter.last_refine_mean
            self.last_suppression_tensor = self.text_guided_adapter.last_refine_tensor
            self.last_suppression_raw_mean = self.text_guided_adapter.last_refine_raw_mean
            self.last_suppression_raw_tensor = self.text_guided_adapter.last_refine_raw_tensor

            memory = rearrange(prompt_memory, "b m c -> m b c")
            query = repeat(query_embed, "b n dim -> n b dim")
            mask_embedding, _ = self.base_model.transformer_decoder(
                tgt=query,
                memory=memory,
                pos=self.base_model.pos_embed,
                memory_key_padding_mask=None,
            )
            mask_embedding = repeat(mask_embedding, "n b dim -> b n dim")

            prompt_embeds = [
                projection(mask_embedding)
                for projection in self.base_model.project_to_decoder_channels
            ]
            decoded = self.base_model.decoder(skips, prompt_embeds)

            if isinstance(decoded, (list, tuple)):
                rectified = []
                for scale_idx, scale_logits in enumerate(decoded):
                    if scale_idx == 0:
                        scale_logits = self.text_guided_adapter.rectify_logits(
                            scale_logits,
                            prompt_text,
                            category_ids=prompt_category_ids,
                        )
                        self.last_candidate_mean = self.text_guided_adapter.last_candidate_mean
                        self.last_risk_mean = self.text_guided_adapter.last_risk_mean
                        self.last_risk_tensor = self.text_guided_adapter.last_risk_tensor
                        self.last_suppression_mean = self.text_guided_adapter.last_refine_mean
                        self.last_suppression_tensor = self.text_guided_adapter.last_refine_tensor
                        self.last_suppression_raw_mean = self.text_guided_adapter.last_refine_raw_mean
                        self.last_suppression_raw_tensor = self.text_guided_adapter.last_refine_raw_tensor
                    rectified.append(scale_logits)
                outs.append(rectified)
            else:
                decoded = self.text_guided_adapter.rectify_logits(
                    decoded,
                    prompt_text,
                    category_ids=prompt_category_ids,
                )
                self.last_candidate_mean = self.text_guided_adapter.last_candidate_mean
                self.last_risk_mean = self.text_guided_adapter.last_risk_mean
                self.last_risk_tensor = self.text_guided_adapter.last_risk_tensor
                self.last_suppression_mean = self.text_guided_adapter.last_refine_mean
                self.last_suppression_tensor = self.text_guided_adapter.last_refine_tensor
                self.last_suppression_raw_mean = self.text_guided_adapter.last_refine_raw_mean
                self.last_suppression_raw_tensor = self.text_guided_adapter.last_refine_raw_tensor
                outs.append(decoded)

        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        if not self.base_model.deep_supervision:
            outs = outs[0]
        return outs
