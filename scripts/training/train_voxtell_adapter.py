import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pydoc
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler
from tqdm import tqdm

PREIMPORT_ROOT = Path(__file__).resolve().parents[2]
PREIMPORT_HF_HOME = PREIMPORT_ROOT / "hf_cache"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HOME", str(PREIMPORT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(PREIMPORT_HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PREIMPORT_HF_HOME / "hub"))

from transformers import AutoModel, AutoTokenizer

ROOT = PREIMPORT_ROOT
VOXTELL_ROOT = ROOT / "VoxTell"
UTILS_ROOT = ROOT / "scripts" / "utils"
for extra_path in [str(VOXTELL_ROOT), str(UTILS_ROOT)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from voxtell.model.v6_adapter_model import VoxTellV64AdapterModel
from voxtell.model.v65_adapter_model import VoxTellV65AdapterModel
from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import (
    build_text_representations,
    collate_text_sequences,
    wrap_with_instruction,
)


TRAIN_LIST = ROOT / "docs" / "plans" / "train_subset_50_cases.txt"
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
TRAIN_DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_train50"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voxtell_adapter_v6_4_train50"

TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
PATCH_SIZE = (192, 192, 192)
TRAINING_STATE_NAME = "training_state.pt"
PROMPT_EMBEDDINGS_NAME = "train_prompt_embeddings.pt"
ADAPTER_STATE_NAME = "adapter_state.pt"
CATEGORY_CODE_MAP = {
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


@dataclass
class TrainSample:
    case_name: str
    finding_idx: int
    prompt: str
    category_code: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_subset_case_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_subset_metadata() -> list[dict]:
    names = set(load_subset_case_names(TRAIN_LIST))
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    selected = [case for case in data["train"] if case["name"] in names]
    missing = sorted(names - {case["name"] for case in selected})
    if missing:
        raise FileNotFoundError(f"Missing train metadata for: {missing}")
    return selected


def flatten_samples(cases: list[dict]) -> list[TrainSample]:
    samples: list[TrainSample] = []
    for case in cases:
        for key in sorted(map(int, case["findings"].keys())):
            idx = str(key)
            samples.append(
                TrainSample(
                    case_name=case["name"],
                    finding_idx=key,
                    prompt=case["findings"][idx],
                    category_code=case.get("categories", {}).get(idx),
                )
            )
    return samples


def build_text_backbone() -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, padding_side="left")
    text_backbone = AutoModel.from_pretrained(TEXT_MODEL_NAME).eval()
    for param in text_backbone.parameters():
        param.requires_grad = False
    return tokenizer, text_backbone


def embed_prompts_cpu(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    text_backbone: AutoModel,
    batch_size: int = 8,
) -> dict[str, dict[str, torch.Tensor]]:
    unique_prompts = list(dict.fromkeys(prompts))
    result: dict[str, dict[str, torch.Tensor]] = {}
    for start in tqdm(range(0, len(unique_prompts), batch_size), desc="Embedding prompts"):
        batch_prompts = unique_prompts[start:start + batch_size]
        wrapped = wrap_with_instruction(batch_prompts)
        tokens = tokenizer(
            wrapped,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = text_backbone(**tokens)
            pooled, token_sequences, token_masks = build_text_representations(
                outputs.last_hidden_state,
                tokens["attention_mask"],
            )
        for prompt, pooled_embedding, token_sequence, token_mask in zip(batch_prompts, pooled, token_sequences, token_masks):
            result[prompt] = {
                "pooled": pooled_embedding.cpu(),
                "tokens": token_sequence,
                "attention_mask": token_mask,
            }
    return result


def get_or_create_text_cache(output_dir: Path, prompts: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    cache_path = output_dir / PROMPT_EMBEDDINGS_NAME
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        valid_cache = bool(cache) and isinstance(next(iter(cache.values())), dict) and "pooled" in next(iter(cache.values()))
        missing = [prompt for prompt in prompts if prompt not in cache] if valid_cache else prompts
        if not missing:
            return cache
    tokenizer, text_backbone = build_text_backbone()
    cache = embed_prompts_cpu(prompts, tokenizer, text_backbone)
    torch.save(cache, cache_path)
    return cache


def resolve_prompt_cache_path(cache_source: Path | None) -> Path | None:
    if cache_source is None:
        return None
    if cache_source.is_dir():
        candidate = cache_source / PROMPT_EMBEDDINGS_NAME
        return candidate if candidate.exists() else None
    return cache_source if cache_source.exists() else None


def resolve_adapter_state_path(adapter_source: Path | None) -> Path | None:
    if adapter_source is None:
        return None
    if adapter_source.is_dir():
        candidate = adapter_source / ADAPTER_STATE_NAME
        return candidate if candidate.exists() else None
    return adapter_source if adapter_source.exists() else None


def infer_resume_step(adapter_source: Path | None, explicit_step: int | None = None) -> int:
    if explicit_step is not None:
        return max(0, int(explicit_step))
    if adapter_source is None or not adapter_source.is_dir():
        return 0
    metrics_path = adapter_source / "train_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return max(0, int(metrics.get("steps_completed", 0)))
        except Exception:
            return 0
    return 0


def get_or_create_text_cache_with_resume(
    output_dir: Path,
    prompts: list[str],
    resume_cache_from: Path | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    cache_path = output_dir / PROMPT_EMBEDDINGS_NAME
    cache: dict[str, dict[str, torch.Tensor]] = {}
    requested_count = len(list(dict.fromkeys(prompts)))
    local_hit_count = 0
    resume_hit_count = 0
    if cache_path.exists():
        loaded = torch.load(cache_path, map_location="cpu")
        if loaded and isinstance(next(iter(loaded.values())), dict) and "pooled" in next(iter(loaded.values())):
            cache.update(loaded)
            local_hit_count = sum(1 for prompt in prompts if prompt in loaded)

    resume_path = resolve_prompt_cache_path(resume_cache_from)
    if resume_path is not None and resume_path != cache_path:
        loaded = torch.load(resume_path, map_location="cpu")
        if loaded and isinstance(next(iter(loaded.values())), dict) and "pooled" in next(iter(loaded.values())):
            resume_hit_count = sum(1 for prompt in prompts if prompt not in cache and prompt in loaded)
            for prompt, entry in loaded.items():
                cache.setdefault(prompt, entry)

    missing = [prompt for prompt in prompts if prompt not in cache]
    total_hit_count = requested_count - len(missing)
    hit_ratio = total_hit_count / max(requested_count, 1)
    resume_label = str(resume_path) if resume_path is not None else "None"
    print(
        f"Prompt cache status: requested={requested_count}, local_hits={local_hit_count}, "
        f"resume_hits={resume_hit_count}, missing={len(missing)}, hit_ratio={hit_ratio:.1%}, "
        f"resume_source={resume_label}"
    )
    if not missing:
        if not cache_path.exists():
            torch.save(cache, cache_path)
        return cache

    tokenizer, text_backbone = build_text_backbone()
    new_entries = embed_prompts_cpu(missing, tokenizer, text_backbone)
    cache.update(new_entries)
    torch.save(cache, cache_path)
    return cache


def build_adapter_network(
    device: torch.device,
    adapter_version: str,
    adapter_hidden_dim: int,
    adapter_insertion_point: str,
    adapter_num_groups: int,
    adapter_risk_groups: int | None,
    adapter_candidate_scale: float,
    adapter_risk_scale: float,
    adapter_gate_cap: float,
    adapter_category_scale: float,
    adapter_refine_scale: float,
) -> nn.Module:
    plans = json.loads((MODEL_DIR / "plans.json").read_text(encoding="utf-8"))
    arch_kwargs = dict(plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for import_key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch_kwargs[import_key] is not None:
            arch_kwargs[import_key] = pydoc.locate(arch_kwargs[import_key])

    base_network = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
        use_token_to_voxel_fusion=False,
        use_scale_aware_prompting=False,
    )
    checkpoint = torch.load(
        MODEL_DIR / "fold_0" / "checkpoint_final.pth",
        map_location="cpu",
        weights_only=False,
    )
    base_network.load_state_dict(checkpoint["network_weights"], strict=False)
    common_kwargs = dict(
        base_model=base_network,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_insertion_point=adapter_insertion_point,
        adapter_num_groups=adapter_num_groups,
        adapter_risk_groups=adapter_risk_groups,
        adapter_candidate_scale=adapter_candidate_scale,
        adapter_risk_scale=adapter_risk_scale,
        adapter_gate_cap=adapter_gate_cap,
        adapter_category_scale=adapter_category_scale,
        use_category_bias=adapter_category_scale > 0.0,
    )
    if adapter_version == "v6_5":
        network = VoxTellV65AdapterModel(
            adapter_refine_scale=adapter_refine_scale,
            **common_kwargs,
        )
    else:
        network = VoxTellV64AdapterModel(**common_kwargs)
    network.to(device)
    apply_training_stage(network, "stage1")
    return network


def compute_schedule_strength(step: int, warmup_steps: int, full_strength_steps: int) -> float:
    if full_strength_steps <= warmup_steps:
        return 1.0
    if step <= warmup_steps:
        return 0.0
    progress = (step - warmup_steps) / max(full_strength_steps - warmup_steps, 1)
    return float(max(0.0, min(1.0, progress)))


def linear_ramp(progress: float, start: float, end: float) -> float:
    progress = max(0.0, min(1.0, float(progress)))
    return float(start + progress * (end - start))


def resolve_stage(step: int, stage1_steps: int, stage2_steps: int) -> str:
    if step < stage1_steps:
        return "stage1"
    if step < stage1_steps + stage2_steps:
        return "stage2"
    return "stage3"


def stage_progress(step: int, stage: str, stage1_steps: int, stage2_steps: int, max_steps: int | None) -> float:
    if stage == "stage1":
        denom = max(stage1_steps, 1)
        offset = step
    elif stage == "stage2":
        denom = max(stage2_steps, 1)
        offset = step - stage1_steps
    else:
        total = max_steps if max_steps is not None else stage1_steps + stage2_steps + 1
        denom = max(total - stage1_steps - stage2_steps, 1)
        offset = step - stage1_steps - stage2_steps
    return max(0.0, min(1.0, offset / denom))


def get_stage_controls(args: argparse.Namespace, step: int) -> dict[str, float | str]:
    stage = resolve_stage(step, args.stage1_steps, args.stage2_steps)
    progress = stage_progress(step, stage, args.stage1_steps, args.stage2_steps, args.max_steps)
    if stage == "stage1":
        return {
            "stage": stage,
            "fusion_strength": linear_ramp(progress, 0.2, 1.0),
            "scale_strength": 0.0,
            "query_strength": 0.0,
            "negative_loss_scale": args.negative_loss_scale * args.stage1_negative_loss_scale_mult,
            "negative_fp_weight": args.negative_fp_weight * args.stage1_negative_fp_weight_mult,
            "fp_aware_penalty_weight": args.fp_aware_penalty_weight * args.stage1_fp_aware_penalty_mult,
            "fusion_reg_positive": args.fusion_reg_weight_positive * args.stage1_fusion_reg_mult,
            "fusion_reg_negative": args.fusion_reg_weight_negative * args.stage1_fusion_reg_mult,
            "query_lr_scale": args.query_lr_scale * args.stage1_query_lr_mult,
        }
    if stage == "stage2":
        return {
            "stage": stage,
            "fusion_strength": 1.0,
            "scale_strength": linear_ramp(progress, 0.0, 1.0),
            "query_strength": linear_ramp(progress, args.stage2_query_start, args.stage2_query_end),
            "negative_loss_scale": args.negative_loss_scale * args.stage2_negative_loss_scale_mult,
            "negative_fp_weight": args.negative_fp_weight * args.stage2_negative_fp_weight_mult,
            "fp_aware_penalty_weight": args.fp_aware_penalty_weight * args.stage2_fp_aware_penalty_mult,
            "fusion_reg_positive": args.fusion_reg_weight_positive * args.stage2_fusion_reg_mult,
            "fusion_reg_negative": args.fusion_reg_weight_negative * args.stage2_fusion_reg_mult,
            "query_lr_scale": args.query_lr_scale * args.stage2_query_lr_mult,
        }
    return {
        "stage": stage,
        "fusion_strength": 1.0,
        "scale_strength": args.stage3_scale_strength,
        "query_strength": linear_ramp(progress, args.stage3_query_start, args.stage3_query_end),
        "negative_loss_scale": args.negative_loss_scale * args.stage3_negative_loss_scale_mult,
        "negative_fp_weight": args.negative_fp_weight * args.stage3_negative_fp_weight_mult,
        "fp_aware_penalty_weight": args.fp_aware_penalty_weight * args.stage3_fp_aware_penalty_mult,
        "fusion_reg_positive": args.fusion_reg_weight_positive * args.stage3_fusion_reg_mult,
        "fusion_reg_negative": args.fusion_reg_weight_negative * args.stage3_fusion_reg_mult,
        "query_lr_scale": args.query_lr_scale * args.stage3_query_lr_mult,
    }


def apply_training_stage(model: nn.Module, stage: str) -> None:
    trainable_keys = {
        "stage1": (
            "text_guided_adapter.to_logit_context",
            "text_guided_adapter.logit_candidate_head",
            "text_guided_adapter.category_embedding",
        ),
        "stage2": (
            "text_guided_adapter.to_logit_context",
            "text_guided_adapter.logit_risk_head",
            "text_guided_adapter.logit_refine_head",
            "text_guided_adapter.category_embedding",
        ),
        "stage3": (
            "text_guided_adapter.to_logit_context",
            "text_guided_adapter.logit_candidate_head",
            "text_guided_adapter.logit_risk_head",
            "text_guided_adapter.logit_refine_head",
            "text_guided_adapter.category_embedding",
        ),
    }[stage]
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in trainable_keys)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float, query_lr_scale: float) -> torch.optim.Optimizer:
    base_params = []
    query_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "project_text_embed" in name or "query_strength_logit" in name:
            query_params.append(param)
        else:
            base_params.append(param)
    param_groups = [{"params": base_params, "lr": lr, "weight_decay": weight_decay}]
    if query_params:
        param_groups.append(
            {
                "params": query_params,
                "lr": lr * query_lr_scale,
                "weight_decay": weight_decay,
            }
        )
    return torch.optim.AdamW(param_groups)


def preprocess_image(image: np.ndarray, normalization: ZScoreNormalization) -> tuple[np.ndarray, tuple, tuple[int, ...]]:
    if image.ndim == 3:
        image = image[None]
    image = image.astype(np.float32, copy=True)
    original_shape = image.shape[1:]
    cropped, _, bbox = crop_to_nonzero(image, None)
    cropped = normalization.run(cropped, None)
    return cropped, bbox, original_shape


def crop_mask_by_bbox(mask_volume: np.ndarray, bbox: tuple) -> np.ndarray:
    x_slice, y_slice, z_slice = bbox
    return mask_volume[:, x_slice[0]:x_slice[1], y_slice[0]:y_slice[1], z_slice[0]:z_slice[1]]


def compute_patch_center(mask: np.ndarray) -> tuple[int, int, int]:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return tuple(dim // 2 for dim in mask.shape)
    center = np.round(foreground.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def clamp_center(center: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
    clamped = []
    for c, dim in zip(center, shape):
        clamped.append(int(max(0, min(c, dim - 1))))
    return tuple(clamped)


def extract_patch(array: np.ndarray, center: tuple[int, int, int], patch_size: tuple[int, int, int], fill_value: float = 0.0) -> np.ndarray:
    spatial = array.shape[-3:]
    starts = [c - size // 2 for c, size in zip(center, patch_size)]
    ends = [start + size for start, size in zip(starts, patch_size)]

    src_slices = []
    dst_slices = []
    for dim, (start, end, size) in enumerate(zip(starts, ends, patch_size)):
        src_start = max(start, 0)
        src_end = min(end, spatial[dim])
        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output = np.full((*array.shape[:-3], *patch_size), fill_value, dtype=array.dtype)
    output[(..., *dst_slices)] = array[(..., *src_slices)]
    return output


def foreground_fraction(mask_patch: np.ndarray) -> float:
    return float(mask_patch.sum() / mask_patch.size) if mask_patch.size else 0.0


def sample_random_center(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(random.randint(0, dim - 1) for dim in shape)


def sample_hard_negative_center(
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    negative_max_fg_fraction: float,
    tries: int = 24,
) -> tuple[int, int, int] | None:
    if mask.sum() == 0:
        return None
    foreground = np.argwhere(mask > 0)
    for _ in range(tries):
        anchor = foreground[random.randrange(len(foreground))]
        jitter = np.array([random.randint(-size // 2, size // 2) for size in patch_size])
        center = clamp_center(tuple((anchor + jitter).tolist()), mask.shape)
        patch = extract_patch(mask, center, patch_size, fill_value=0.0)
        frac = foreground_fraction(patch)
        if 0.0 < frac <= negative_max_fg_fraction:
            return center
    return None


def choose_patch_center(
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
    positive_prob: float,
    hard_negative_prob: float,
    negative_max_fg_fraction: float,
    force_positive: bool,
) -> tuple[tuple[int, int, int], str]:
    has_fg = bool(mask.sum() > 0)
    if force_positive and has_fg:
        return compute_patch_center(mask), "positive_warmup"
    if not has_fg:
        return sample_random_center(mask.shape), "random_negative"

    roll = random.random()
    if roll < positive_prob:
        return compute_patch_center(mask), "positive"
    if roll < positive_prob + hard_negative_prob:
        hard_center = sample_hard_negative_center(mask, patch_size, negative_max_fg_fraction, tries=24)
        if hard_center is not None:
            return hard_center, "hard_negative"
    return sample_random_center(mask.shape), "random_negative"


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight))
    return (loss * weights).mean()


def false_positive_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs * (1.0 - targets)).mean()


def predicted_foreground_fraction(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float().mean()


def suppression_bce_loss(
    suppression_tensor: torch.Tensor | None,
    target_value: float,
) -> torch.Tensor | None:
    if suppression_tensor is None:
        return None
    target = torch.full_like(suppression_tensor, fill_value=target_value, dtype=torch.float32)
    suppression_tensor = suppression_tensor.float()
    device_type = suppression_tensor.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return F.binary_cross_entropy(suppression_tensor, target)


def compute_suppression_target(
    args: argparse.Namespace,
    sample_mode: str,
    is_negative_patch: bool,
    patch_fg_fraction: float,
    pred_fg_fraction: float,
    fp_penalty: float,
) -> float:
    if args.suppression_target_mode == "fp_aware_v5_1":
        if sample_mode in {"positive", "positive_warmup"}:
            fg_term = min(patch_fg_fraction / max(args.suppression_target_positive_fg_norm, 1e-6), 1.0)
            target = args.suppression_target_positive_base + args.suppression_target_positive_fg_scale * fg_term
        elif sample_mode == "hard_negative":
            target = (
                args.suppression_target_hard_negative_base
                + args.suppression_target_hard_negative_fg_scale * min(
                    patch_fg_fraction / max(args.suppression_target_hard_negative_fg_norm, 1e-6), 1.0
                )
                - args.suppression_target_hard_negative_pred_scale * pred_fg_fraction
                - args.suppression_target_hard_negative_fp_scale * fp_penalty
            )
        else:
            target = (
                args.suppression_target_random_negative_base
                - args.suppression_target_random_negative_pred_scale * pred_fg_fraction
                - args.suppression_target_random_negative_fp_scale * fp_penalty
            )
        return float(max(0.0, min(1.0, target)))
    if args.suppression_target_mode == "soft_by_sample_mode":
        if sample_mode == "hard_negative":
            return float(args.suppression_target_hard_negative)
        if sample_mode == "random_negative":
            return float(args.suppression_target_random_negative)
        return float(args.suppression_target_positive)
    return 0.0 if is_negative_patch else 1.0


def compute_suppression_weight(args: argparse.Namespace, sample_mode: str, is_negative_patch: bool) -> float:
    base_weight = float(args.suppression_loss_weight)
    if base_weight <= 0.0:
        return 0.0
    if args.suppression_target_mode == "fp_aware_v5_1":
        if sample_mode in {"positive", "positive_warmup"}:
            return base_weight * float(args.suppression_weight_positive)
        if sample_mode == "hard_negative":
            return base_weight * float(args.suppression_weight_hard_negative)
        return base_weight * float(args.suppression_weight_random_negative)
    return base_weight


def compute_fp_aware_penalty(
    args: argparse.Namespace,
    suppression_tensor: torch.Tensor | None,
    sample_mode: str,
    is_negative_patch: bool,
    fp_penalty: torch.Tensor,
    pred_fg_fraction: torch.Tensor,
) -> torch.Tensor | None:
    if suppression_tensor is None or not is_negative_patch or args.fp_aware_penalty_weight <= 0.0:
        return None
    sample_scale = (
        args.fp_aware_penalty_hard_negative_scale
        if sample_mode == "hard_negative"
        else args.fp_aware_penalty_random_negative_scale
    )
    suppression_mean = suppression_tensor.mean()
    driving_signal = (
        args.fp_aware_penalty_fp_scale * fp_penalty.detach()
        + args.fp_aware_penalty_pred_scale * pred_fg_fraction.detach()
    )
    return args.fp_aware_penalty_weight * sample_scale * suppression_mean * driving_signal


def load_case_arrays(case_name: str, normalization: ZScoreNormalization) -> tuple[np.ndarray, np.ndarray]:
    image_path = TRAIN_DATASET_DIR / "images_flat" / case_name
    seg_path = TRAIN_DATASET_DIR / "segmentations" / case_name
    image = np.asarray(nib.load(str(image_path)).dataobj).astype(np.float32)
    seg = np.asarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)
    if seg.ndim == 3:
        seg = seg[None]
    preprocessed, bbox, _ = preprocess_image(image, normalization)
    cropped_seg = crop_mask_by_bbox(seg, bbox).astype(np.float32)
    return preprocessed, cropped_seg


def iter_case_findings(case: dict, max_findings_per_case: int | None) -> Iterable[tuple[int, str, str | None]]:
    keys = sorted(map(int, case["findings"].keys()))
    if max_findings_per_case is not None:
        keys = keys[:max_findings_per_case]
    for key in keys:
        idx = str(key)
        yield key, case["findings"][idx], case.get("categories", {}).get(idx)


def map_category_code(category_code: str | None) -> int:
    if category_code is None:
        return 0
    return int(CATEGORY_CODE_MAP.get(category_code, 0))


def save_training_state(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    state: dict,
    include_training_state: bool = False,
) -> None:
    final_step = int(state.get("step", 0) or 0)
    final_milestone_path = output_dir / f"adapter_step_{final_step:04d}.pt"
    final_adapter_path = output_dir / ADAPTER_STATE_NAME
    if final_step > 0 and final_milestone_path.exists():
        if final_adapter_path.exists():
            final_adapter_path.unlink(missing_ok=True)
        final_milestone_path.replace(final_adapter_path)
    else:
        torch.save(model.state_dict(), final_adapter_path)
    if not include_training_state:
        return
    adapter_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "state": state,
    }
    torch.save(adapter_state, output_dir / TRAINING_STATE_NAME)


def save_milestone_checkpoint(output_dir: Path, model: nn.Module, step: int, keep_last: int) -> None:
    checkpoint_path = output_dir / f"adapter_step_{step:04d}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("adapter_step_*.pt"))
    overflow = len(checkpoints) - keep_last
    for stale_path in checkpoints[:max(0, overflow)]:
        stale_path.unlink(missing_ok=True)


def summarize_loss(loss_history: list[dict]) -> dict[str, float | None]:
    if not loss_history:
        return {"final_loss": None, "best_loss": None}
    losses = [entry["loss"] for entry in loss_history]
    return {"final_loss": float(losses[-1]), "best_loss": float(min(losses))}


def summarize_training_diagnostics(loss_history: list[dict]) -> dict[str, float | int | None]:
    if not loss_history:
        return {}
    zero_entries = [entry for entry in loss_history if entry["loss"] <= 0.0]
    negative_entries = [entry for entry in loss_history if entry["negative_scaled"]]
    positive_entries = [entry for entry in loss_history if not entry["negative_scaled"]]
    hard_negative_entries = [entry for entry in loss_history if entry["sample_mode"] == "hard_negative"]
    random_negative_entries = [entry for entry in loss_history if entry["sample_mode"] == "random_negative"]

    def mean_or_none(values: list[float | None]) -> float | None:
        clean = [float(v) for v in values if v is not None]
        return float(sum(clean) / len(clean)) if clean else None

    return {
        "zero_loss_count": len(zero_entries),
        "zero_loss_ratio": float(len(zero_entries) / len(loss_history)),
        "zero_loss_neg_count": sum(1 for entry in zero_entries if entry["negative_scaled"]),
        "hard_negative_count": len(hard_negative_entries),
        "random_negative_count": len(random_negative_entries),
        "positive_count": len(positive_entries),
        "avg_patch_fg_positive": mean_or_none([entry.get("patch_fg_fraction") for entry in positive_entries]),
        "avg_patch_fg_negative": mean_or_none([entry.get("patch_fg_fraction") for entry in negative_entries]),
        "avg_fp_penalty_positive": mean_or_none([entry.get("fp_penalty") for entry in positive_entries]),
        "avg_fp_penalty_negative": mean_or_none([entry.get("fp_penalty") for entry in negative_entries]),
        "avg_pred_fg_positive": mean_or_none([entry.get("pred_fg_fraction") for entry in positive_entries]),
        "avg_pred_fg_negative": mean_or_none([entry.get("pred_fg_fraction") for entry in negative_entries]),
        "avg_fp_aware_penalty_positive": mean_or_none([entry.get("fp_aware_penalty") for entry in positive_entries]),
        "avg_fp_aware_penalty_negative": mean_or_none([entry.get("fp_aware_penalty") for entry in negative_entries]),
        "suppression_mean_pos_avg": mean_or_none([entry.get("suppression_mean") for entry in positive_entries]),
        "suppression_mean_neg_avg": mean_or_none([entry.get("suppression_mean") for entry in negative_entries]),
        "avg_suppression_loss_positive": mean_or_none([entry.get("suppression_loss") for entry in positive_entries]),
        "avg_suppression_loss_negative": mean_or_none([entry.get("suppression_loss") for entry in negative_entries]),
        "avg_suppression_target_positive": mean_or_none([entry.get("suppression_target") for entry in positive_entries]),
        "avg_suppression_target_negative": mean_or_none([entry.get("suppression_target") for entry in negative_entries]),
        "avg_fusion_strength": mean_or_none([entry.get("fusion_strength") for entry in loss_history]),
        "avg_scale_strength": mean_or_none([entry.get("scale_strength") for entry in loss_history]),
        "avg_query_strength": mean_or_none([entry.get("query_strength") for entry in loss_history]),
        "avg_query_scale": mean_or_none([entry.get("query_scale") for entry in loss_history]),
        "avg_fusion_delta_positive": mean_or_none([entry.get("fusion_delta") for entry in positive_entries]),
        "avg_fusion_delta_negative": mean_or_none([entry.get("fusion_delta") for entry in negative_entries]),
        "avg_fusion_gate_positive": mean_or_none([entry.get("fusion_gate") for entry in positive_entries]),
        "avg_fusion_gate_negative": mean_or_none([entry.get("fusion_gate") for entry in negative_entries]),
        "avg_fusion_reg_positive": mean_or_none([entry.get("fusion_reg") for entry in positive_entries]),
        "avg_fusion_reg_negative": mean_or_none([entry.get("fusion_reg") for entry in negative_entries]),
    }


def train(args: argparse.Namespace) -> dict:
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
        os.environ["HF_HUB_CACHE"] = str(args.hf_home / "hub")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_subset_metadata()
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    samples = flatten_samples(cases)
    all_prompts = [sample.prompt for sample in samples]
    text_cache = get_or_create_text_cache_with_resume(
        output_dir,
        all_prompts,
        resume_cache_from=args.resume_prompt_cache_from,
    )

    normalization = ZScoreNormalization(intensityproperties={})
    model = build_adapter_network(
        device,
        adapter_version=args.adapter_version,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_insertion_point=args.adapter_insertion_point,
        adapter_num_groups=args.adapter_num_groups,
        adapter_risk_groups=args.adapter_risk_groups,
        adapter_candidate_scale=args.adapter_candidate_scale,
        adapter_risk_scale=args.adapter_risk_scale,
        adapter_gate_cap=args.adapter_gate_cap,
        adapter_category_scale=args.adapter_category_scale,
        adapter_refine_scale=args.adapter_refine_scale,
    )
    resume_adapter_path = resolve_adapter_state_path(args.resume_adapter_from)
    if resume_adapter_path is not None:
        resume_state = torch.load(resume_adapter_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_state, strict=False)
        print(f"Resumed adapter weights from: {resume_adapter_path}")
    model.train()
    step = infer_resume_step(args.resume_adapter_from, args.resume_step)
    initial_stage_controls = get_stage_controls(args, step)
    current_stage = str(initial_stage_controls["stage"])
    apply_training_stage(model, current_stage)
    optimizer = make_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        query_lr_scale=float(initial_stage_controls["query_lr_scale"]),
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    print(f"Train cases: {len(cases)}")
    print(f"Train findings: {len(samples)}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    if step > 0:
        print(f"Resume step: {step}")

    loss_history: list[dict] = []
    for epoch in range(args.epochs):
        random.shuffle(cases)
        epoch_bar = tqdm(cases, desc=f"Epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for case in epoch_bar:
            image_arr, seg_arr = load_case_arrays(case["name"], normalization)
            finding_items = list(iter_case_findings(case, args.max_findings_per_case))
            random.shuffle(finding_items)

            for finding_idx, prompt, category_code in finding_items:
                stage_controls = get_stage_controls(args, step)
                stage_name = str(stage_controls["stage"])
                if stage_name != current_stage:
                    current_stage = stage_name
                    apply_training_stage(model, current_stage)
                    optimizer = make_optimizer(
                        model,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        query_lr_scale=float(stage_controls["query_lr_scale"]),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    print(f"Switched training stage -> {current_stage} at step {step}")

                mask_arr = seg_arr[finding_idx]
                center, sample_mode = choose_patch_center(
                    mask_arr,
                    PATCH_SIZE,
                    positive_prob=args.positive_patch_prob,
                    hard_negative_prob=args.hard_negative_patch_prob,
                    negative_max_fg_fraction=args.negative_max_fg_fraction,
                    force_positive=step < args.negative_sampling_after_steps,
                )
                image_patch = extract_patch(image_arr, center, PATCH_SIZE, fill_value=0.0)
                mask_patch = extract_patch(mask_arr, center, PATCH_SIZE, fill_value=0.0)

                image_tensor = torch.from_numpy(image_patch[None]).to(device=device, dtype=torch.float32)
                target_tensor = torch.from_numpy(mask_patch[None, None]).to(device=device, dtype=torch.float32)
                prompt_entry = text_cache[prompt]
                text_embedding = prompt_entry["pooled"].view(1, 1, -1).to(device=device, dtype=torch.float32)
                category_tensor = torch.tensor(
                    [[map_category_code(category_code)]],
                    device=device,
                    dtype=torch.long,
                )
                text_token_embeddings, text_attention_mask = collate_text_sequences(
                    [prompt_entry["tokens"]],
                    [prompt_entry["attention_mask"]],
                    device=device,
                    dtype=torch.float32,
                )
                patch_fg_fraction = foreground_fraction(mask_patch)
                is_negative_patch = sample_mode in {"random_negative", "hard_negative"}
                fusion_strength = float(stage_controls["fusion_strength"])
                scale_strength = float(stage_controls["scale_strength"])
                query_strength = float(stage_controls["query_strength"])
                model.set_fusion_schedule(fusion_strength=fusion_strength, scale_strength=scale_strength)
                model.set_query_schedule(query_strength=query_strength)

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda" and not args.no_amp
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(
                        image_tensor,
                        text_embedding,
                        category_ids=category_tensor,
                        text_token_embeddings=text_token_embeddings,
                        text_attention_mask=text_attention_mask,
                    )
                    neg_bce_weight = args.negative_bce_weight if is_negative_patch else 1.0
                    neg_loss_scale = float(stage_controls["negative_loss_scale"]) if is_negative_patch else 1.0
                    fp_weight = float(stage_controls["negative_fp_weight"]) if is_negative_patch else args.fp_weight
                    bce = weighted_bce_with_logits(logits, target_tensor, negative_weight=neg_bce_weight)
                    dice = dice_loss_from_logits(logits, target_tensor)
                    fp_penalty = false_positive_penalty(logits, target_tensor)
                    pred_fg_fraction_tensor = predicted_foreground_fraction(logits, threshold=args.pred_fg_threshold)
                    seg_loss = neg_loss_scale * (args.bce_weight * bce + args.dice_weight * dice) + fp_weight * fp_penalty
                    suppression_tensor = getattr(model, "last_suppression_tensor", None)
                    suppression_target = compute_suppression_target(
                        args=args,
                        sample_mode=sample_mode,
                        is_negative_patch=is_negative_patch,
                        patch_fg_fraction=patch_fg_fraction,
                        pred_fg_fraction=float(pred_fg_fraction_tensor.detach().cpu()),
                        fp_penalty=float(fp_penalty.detach().cpu()),
                    )
                    suppression_weight = compute_suppression_weight(
                        args=args,
                        sample_mode=sample_mode,
                        is_negative_patch=is_negative_patch,
                    )
                    suppression_loss = suppression_bce_loss(
                        suppression_tensor=suppression_tensor,
                        target_value=suppression_target,
                    )
                    fusion_delta = getattr(model, "last_token_fusion_delta_mean", None)
                    fusion_gate = getattr(model, "last_token_gate_mean", None)
                    fusion_reg = torch.zeros((), device=device, dtype=seg_loss.dtype)
                    if fusion_delta is not None:
                        fusion_reg = torch.as_tensor(
                            fusion_delta * (
                                float(stage_controls["fusion_reg_negative"])
                                if is_negative_patch
                                else float(stage_controls["fusion_reg_positive"])
                            ),
                            device=device,
                            dtype=seg_loss.dtype,
                        )
                pred_fg_fraction = float(pred_fg_fraction_tensor.detach().cpu())
                fp_aware_penalty = compute_fp_aware_penalty(
                    args=args,
                    suppression_tensor=suppression_tensor,
                    sample_mode=sample_mode,
                    is_negative_patch=is_negative_patch,
                    fp_penalty=fp_penalty,
                    pred_fg_fraction=pred_fg_fraction_tensor,
                )
                if fp_aware_penalty is None:
                    fp_aware_penalty = torch.zeros((), device=device, dtype=seg_loss.dtype)
                else:
                    fp_aware_penalty = fp_aware_penalty.to(device=device, dtype=seg_loss.dtype)
                stage_fp_scale = float(stage_controls["fp_aware_penalty_weight"]) / max(args.fp_aware_penalty_weight, 1e-6) if args.fp_aware_penalty_weight > 0 else 0.0
                fp_aware_penalty = fp_aware_penalty * stage_fp_scale
                suppression_term = torch.zeros((), device=device, dtype=seg_loss.dtype)
                if suppression_loss is not None:
                    suppression_term = suppression_loss.to(device=device, dtype=seg_loss.dtype) * suppression_weight
                loss = seg_loss + suppression_term + fp_aware_penalty + fusion_reg

                loss_item = float(loss.detach().cpu())
                easy_empty_negative = (
                    args.skip_easy_empty_negatives
                    and is_negative_patch
                    and patch_fg_fraction == 0.0
                    and loss_item <= args.easy_negative_loss_threshold
                )
                if easy_empty_negative:
                    continue

                scaler.scale(loss / args.grad_accum_steps).backward()
                step += 1
                loss_history.append(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "case_name": case["name"],
                        "finding_idx": finding_idx,
                        "category_code": category_code,
                        "prompt": prompt,
                        "sample_mode": sample_mode,
                        "stage": current_stage,
                        "patch_fg_fraction": patch_fg_fraction,
                        "loss": loss_item,
                        "seg_loss": float(seg_loss.detach().cpu()),
                        "bce": float(bce.detach().cpu()),
                        "dice_loss": float(dice.detach().cpu()),
                        "fp_penalty": float(fp_penalty.detach().cpu()),
                        "pred_fg_fraction": pred_fg_fraction,
                        "fp_aware_penalty": float(fp_aware_penalty.detach().cpu()),
                        "suppression_loss": float(suppression_term.detach().cpu()),
                        "suppression_target": suppression_target,
                        "suppression_mean": getattr(model, "last_suppression_mean", None),
                        "fusion_strength": fusion_strength,
                        "scale_strength": scale_strength,
                        "query_strength": query_strength,
                        "query_scale": getattr(model, "last_query_scale", None),
                        "fusion_delta": fusion_delta,
                        "fusion_gate": fusion_gate,
                        "fusion_reg": float(fusion_reg.detach().cpu()),
                        "is_zero_loss": bool(loss_item <= 0.0),
                        "negative_scaled": bool(is_negative_patch),
                    }
                )
                epoch_bar.set_postfix(loss=f"{loss_item:.4f}", step=step)

                if step % args.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if args.save_every_steps and step % args.save_every_steps == 0:
                    save_milestone_checkpoint(
                        output_dir=output_dir,
                        model=model,
                        step=step,
                        keep_last=args.max_milestone_checkpoints,
                    )

                if args.max_steps is not None and step >= args.max_steps:
                    break

            del image_arr, seg_arr
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.max_steps is not None and step >= args.max_steps:
                break

        if step % args.grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        save_training_state(
            output_dir,
            model,
            optimizer,
            scaler,
            {
                "current_epoch": epoch + 1,
                "step": step,
                "loss_history": loss_history,
            },
            include_training_state=args.save_training_state_final,
        )

        if args.max_steps is not None and step >= args.max_steps:
            break

    metrics = {
        "epochs": args.epochs,
        "steps_completed": step,
        "train_cases": len(cases),
        "train_findings": len(samples),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "loss_history": loss_history,
        "diagnostics": summarize_training_diagnostics(loss_history),
        **summarize_loss(loss_history),
    }
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return {
        "output_dir": str(output_dir),
        "steps_completed": step,
        "train_cases": len(cases),
        "train_findings": len(samples),
        **summarize_loss(loss_history),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal adapter training for VoxTell on the 50-case train subset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-home", type=Path, default=ROOT / "hf_cache")
    parser.add_argument("--resume-prompt-cache-from", type=Path, default=None)
    parser.add_argument("--resume-adapter-from", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-findings-per-case", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--max-milestone-checkpoints", type=int, default=3)
    parser.add_argument("--save-training-state-final", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--fp-weight", type=float, default=0.05)
    parser.add_argument("--negative-fp-weight", type=float, default=0.35)
    parser.add_argument("--negative-bce-weight", type=float, default=4.0)
    parser.add_argument("--negative-loss-scale", type=float, default=2.5)
    parser.add_argument("--suppression-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--suppression-target-mode",
        choices=["binary", "soft_by_sample_mode", "fp_aware_v5_1"],
        default="binary",
    )
    parser.add_argument("--suppression-target-positive", type=float, default=1.0)
    parser.add_argument("--suppression-target-hard-negative", type=float, default=0.4)
    parser.add_argument("--suppression-target-random-negative", type=float, default=0.0)
    parser.add_argument("--suppression-target-positive-base", type=float, default=0.75)
    parser.add_argument("--suppression-target-positive-fg-scale", type=float, default=0.20)
    parser.add_argument("--suppression-target-positive-fg-norm", type=float, default=0.02)
    parser.add_argument("--suppression-target-hard-negative-base", type=float, default=0.25)
    parser.add_argument("--suppression-target-hard-negative-fg-scale", type=float, default=0.10)
    parser.add_argument("--suppression-target-hard-negative-fg-norm", type=float, default=0.002)
    parser.add_argument("--suppression-target-hard-negative-pred-scale", type=float, default=0.75)
    parser.add_argument("--suppression-target-hard-negative-fp-scale", type=float, default=4.0)
    parser.add_argument("--suppression-target-random-negative-base", type=float, default=0.05)
    parser.add_argument("--suppression-target-random-negative-pred-scale", type=float, default=0.50)
    parser.add_argument("--suppression-target-random-negative-fp-scale", type=float, default=2.5)
    parser.add_argument("--suppression-weight-positive", type=float, default=0.05)
    parser.add_argument("--suppression-weight-hard-negative", type=float, default=0.35)
    parser.add_argument("--suppression-weight-random-negative", type=float, default=0.20)
    parser.add_argument("--fp-aware-penalty-weight", type=float, default=0.0)
    parser.add_argument("--fp-aware-penalty-hard-negative-scale", type=float, default=1.5)
    parser.add_argument("--fp-aware-penalty-random-negative-scale", type=float, default=0.75)
    parser.add_argument("--fp-aware-penalty-fp-scale", type=float, default=1.0)
    parser.add_argument("--fp-aware-penalty-pred-scale", type=float, default=0.5)
    parser.add_argument("--pred-fg-threshold", type=float, default=0.5)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-version", choices=["v6_4", "v6_5"], default="v6_4")
    parser.add_argument("--adapter-insertion-point", choices=["pre_decoder", "post_decoder"], default="pre_decoder")
    parser.add_argument("--adapter-num-groups", type=int, default=8)
    parser.add_argument("--adapter-risk-groups", type=int, default=None)
    parser.add_argument("--adapter-candidate-scale", type=float, default=0.1)
    parser.add_argument("--adapter-risk-scale", type=float, default=0.08)
    parser.add_argument("--adapter-gate-cap", type=float, default=0.25)
    parser.add_argument("--adapter-category-scale", type=float, default=0.2)
    parser.add_argument("--adapter-refine-scale", type=float, default=0.035)
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=100)
    parser.add_argument("--stage1-negative-loss-scale-mult", type=float, default=0.8)
    parser.add_argument("--stage1-negative-fp-weight-mult", type=float, default=0.75)
    parser.add_argument("--stage1-fp-aware-penalty-mult", type=float, default=0.0)
    parser.add_argument("--stage1-fusion-reg-mult", type=float, default=1.0)
    parser.add_argument("--stage1-query-lr-mult", type=float, default=0.0)
    parser.add_argument("--stage2-query-start", type=float, default=0.1)
    parser.add_argument("--stage2-query-end", type=float, default=0.45)
    parser.add_argument("--stage2-negative-loss-scale-mult", type=float, default=1.0)
    parser.add_argument("--stage2-negative-fp-weight-mult", type=float, default=0.9)
    parser.add_argument("--stage2-fp-aware-penalty-mult", type=float, default=0.55)
    parser.add_argument("--stage2-fusion-reg-mult", type=float, default=0.8)
    parser.add_argument("--stage2-query-lr-mult", type=float, default=1.0)
    parser.add_argument("--stage3-query-start", type=float, default=0.25)
    parser.add_argument("--stage3-query-end", type=float, default=0.4)
    parser.add_argument("--stage3-scale-strength", type=float, default=0.4)
    parser.add_argument("--stage3-negative-loss-scale-mult", type=float, default=0.9)
    parser.add_argument("--stage3-negative-fp-weight-mult", type=float, default=1.0)
    parser.add_argument("--stage3-fp-aware-penalty-mult", type=float, default=0.7)
    parser.add_argument("--stage3-fusion-reg-mult", type=float, default=0.6)
    parser.add_argument("--stage3-query-lr-mult", type=float, default=1.0)
    parser.add_argument("--fusion-warmup-steps", type=int, default=20)
    parser.add_argument("--fusion-full-strength-steps", type=int, default=120)
    parser.add_argument("--scale-warmup-steps", type=int, default=40)
    parser.add_argument("--scale-full-strength-steps", type=int, default=180)
    parser.add_argument("--query-warmup-steps", type=int, default=40)
    parser.add_argument("--query-full-strength-steps", type=int, default=120)
    parser.add_argument("--query-lr-scale", type=float, default=0.2)
    parser.add_argument("--fusion-reg-weight-positive", type=float, default=0.01)
    parser.add_argument("--fusion-reg-weight-negative", type=float, default=0.04)
    parser.add_argument("--positive-patch-prob", type=float, default=0.40)
    parser.add_argument("--hard-negative-patch-prob", type=float, default=0.50)
    parser.add_argument("--negative-max-fg-fraction", type=float, default=0.001)
    parser.add_argument("--negative-sampling-after-steps", type=int, default=30)
    parser.add_argument("--skip-easy-empty-negatives", action="store_true")
    parser.add_argument("--easy-negative-loss-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.output_dir = ROOT / "outputs" / "voxtell_adapter_smoke"
        args.epochs = 1
        args.max_cases = 3
        args.max_findings_per_case = 2
        args.max_steps = 4
        args.save_every_steps = 2
    if not args.smoke and not args.skip_easy_empty_negatives:
        args.skip_easy_empty_negatives = True
    if args.suppression_target_mode == "fp_aware_v5_1":
        args.suppression_loss_weight = 0.0
        args.fp_aware_penalty_weight = max(args.fp_aware_penalty_weight, 0.35)
        args.negative_fp_weight = max(args.negative_fp_weight, 0.5)
        args.negative_loss_scale = max(args.negative_loss_scale, 3.0)
    return args


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
