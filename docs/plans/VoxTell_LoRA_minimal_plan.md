# VoxTell LoRA Minimal Plan

Date: 2026-04-13

## Goal

Add a minimal, low-risk fine-tuning path for `VoxTell` without depending on an official training script.

## Current Findings

- `VoxTell` model structure is visible in local source code and is not a black box.
- Actual inference path:
  `Qwen text backbone -> project_text_embed -> transformer_decoder -> project_to_decoder_channels -> decoder`
- The most likely adaptation bottleneck is text-to-mask alignment, not the CNN backbone.
- First LoRA experiment should avoid full `Qwen` fine-tuning and avoid touching 3D conv blocks.

## Recommended LoRA Scope

Freeze:
- `text_backbone (Qwen)`
- `encoder`
- convolutional `decoder`

Apply LoRA to:
- `project_text_embed`
- `transformer_decoder`
- `project_to_decoder_channels`

## Concrete Target Modules

- `project_text_embed.0`
- `project_text_embed.2`
- `transformer_decoder.layers.{0..5}.linear1`
- `transformer_decoder.layers.{0..5}.linear2`
- `transformer_decoder.layers.{0..5}.self_attn.out_proj`
- `transformer_decoder.layers.{0..5}.multihead_attn.out_proj`
- `project_to_decoder_channels.{0..4}.0`
- `project_to_decoder_channels.{0..4}.2`

Do not target `MultiheadAttention.in_proj_weight` in Phase A. PyTorch packs `q/k/v` together there, so off-the-shelf `PEFT` is less clean on that weight. If Phase A works, Phase B can refactor attention into explicit `q_proj / k_proj / v_proj`.

## Training Recipe

- Initialize from existing `checkpoint_final.pth`
- Use current text prompts first
- Loss: `Dice + BCEWithLogits`
- Optimizer: `AdamW` on LoRA params only
- Start with a small train subset for smoke test
- Validate on the same public `val` split against current `raw` baseline

## Why This Is the Lowest-Risk Path

- No need to reproduce official VoxTell training
- No need to full-finetune `Qwen 4B`
- Keeps trainable scope focused on prompt-to-mask alignment
- Rollback cost is low

## Next Step

Build a standalone fine-tuning script that reuses current model loading code, injects LoRA into the listed modules, and trains on CT plus finding-mask pairs.
