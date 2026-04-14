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
from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


TRAIN_LIST = ROOT / "docs" / "plans" / "train_subset_50_cases.txt"
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
TRAIN_DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_train50"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voxtell_adapter_train50"

TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
PATCH_SIZE = (192, 192, 192)
TRAINING_STATE_NAME = "training_state.pt"
PROMPT_EMBEDDINGS_NAME = "train_prompt_embeddings.pt"
ADAPTER_STATE_NAME = "adapter_state.pt"


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
) -> dict[str, torch.Tensor]:
    unique_prompts = list(dict.fromkeys(prompts))
    result: dict[str, torch.Tensor] = {}
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
            embeddings = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"]).float()
        for prompt, embedding in zip(batch_prompts, embeddings):
            result[prompt] = embedding.cpu()
    return result


def get_or_create_text_cache(output_dir: Path, prompts: list[str]) -> dict[str, torch.Tensor]:
    cache_path = output_dir / PROMPT_EMBEDDINGS_NAME
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        missing = [prompt for prompt in prompts if prompt not in cache]
        if not missing:
            return cache
    tokenizer, text_backbone = build_text_backbone()
    cache = embed_prompts_cpu(prompts, tokenizer, text_backbone)
    torch.save(cache, cache_path)
    return cache


def build_adapter_network(
    device: torch.device,
    adapter_hidden_dim: int,
    adapter_insertion_point: str,
    adapter_num_groups: int,
    adapter_residual_scale: float,
    adapter_gate_cap: float,
) -> nn.Module:
    plans = json.loads((MODEL_DIR / "plans.json").read_text(encoding="utf-8"))
    arch_kwargs = dict(plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for import_key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch_kwargs[import_key] is not None:
            arch_kwargs[import_key] = pydoc.locate(arch_kwargs[import_key])

    network = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
        use_text_guided_adapter=True,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_insertion_point=adapter_insertion_point,
        adapter_num_groups=adapter_num_groups,
        adapter_residual_scale=adapter_residual_scale,
        adapter_gate_cap=adapter_gate_cap,
    )
    checkpoint = torch.load(
        MODEL_DIR / "fold_0" / "checkpoint_final.pth",
        map_location="cpu",
        weights_only=False,
    )
    network.load_state_dict(checkpoint["network_weights"], strict=False)
    network.to(device)
    for name, param in network.named_parameters():
        param.requires_grad = "text_guided_adapter" in name
    return network


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


def save_training_state(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    state: dict,
) -> None:
    adapter_state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "state": state,
    }
    torch.save(adapter_state, output_dir / TRAINING_STATE_NAME)
    torch.save(model.state_dict(), output_dir / ADAPTER_STATE_NAME)


def summarize_loss(loss_history: list[dict]) -> dict[str, float | None]:
    if not loss_history:
        return {"final_loss": None, "best_loss": None}
    losses = [entry["loss"] for entry in loss_history]
    return {"final_loss": float(losses[-1]), "best_loss": float(min(losses))}


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
    text_cache = get_or_create_text_cache(output_dir, [sample.prompt for sample in samples])

    normalization = ZScoreNormalization(intensityproperties={})
    model = build_adapter_network(
        device,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_insertion_point=args.adapter_insertion_point,
        adapter_num_groups=args.adapter_num_groups,
        adapter_residual_scale=args.adapter_residual_scale,
        adapter_gate_cap=args.adapter_gate_cap,
    )
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Using device: {device}")
    print(f"Train cases: {len(cases)}")
    print(f"Train findings: {len(samples)}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    step = 0
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
                text_embedding = text_cache[prompt].view(1, 1, -1).to(device=device, dtype=torch.float32)

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device.type == "cuda" and not args.no_amp
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(image_tensor, text_embedding)
                    is_negative_patch = sample_mode in {"random_negative", "hard_negative"}
                    neg_bce_weight = args.negative_bce_weight if is_negative_patch else 1.0
                    neg_loss_scale = args.negative_loss_scale if is_negative_patch else 1.0
                    fp_weight = args.negative_fp_weight if is_negative_patch else args.fp_weight
                    bce = weighted_bce_with_logits(logits, target_tensor, negative_weight=neg_bce_weight)
                    dice = dice_loss_from_logits(logits, target_tensor)
                    fp_penalty = false_positive_penalty(logits, target_tensor)
                    loss = neg_loss_scale * (args.bce_weight * bce + args.dice_weight * dice) + fp_weight * fp_penalty

                scaler.scale(loss / args.grad_accum_steps).backward()
                step += 1
                loss_item = float(loss.detach().cpu())
                loss_history.append(
                    {
                        "step": step,
                        "epoch": epoch + 1,
                        "case_name": case["name"],
                        "finding_idx": finding_idx,
                        "category_code": category_code,
                        "prompt": prompt,
                        "sample_mode": sample_mode,
                        "patch_fg_fraction": foreground_fraction(mask_patch),
                        "loss": loss_item,
                        "bce": float(bce.detach().cpu()),
                        "dice_loss": float(dice.detach().cpu()),
                        "fp_penalty": float(fp_penalty.detach().cpu()),
                        "negative_scaled": bool(is_negative_patch),
                    }
                )
                epoch_bar.set_postfix(loss=f"{loss_item:.4f}", step=step)

                if step % args.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if args.save_every_steps and step % args.save_every_steps == 0:
                    save_training_state(
                        output_dir,
                        model,
                        optimizer,
                        scaler,
                        {
                            "current_epoch": epoch,
                            "step": step,
                            "loss_history": loss_history,
                        },
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-findings-per-case", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--fp-weight", type=float, default=0.05)
    parser.add_argument("--negative-fp-weight", type=float, default=0.25)
    parser.add_argument("--negative-bce-weight", type=float, default=4.0)
    parser.add_argument("--negative-loss-scale", type=float, default=2.0)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-insertion-point", choices=["pre_decoder", "post_decoder"], default="pre_decoder")
    parser.add_argument("--adapter-num-groups", type=int, default=8)
    parser.add_argument("--adapter-residual-scale", type=float, default=0.1)
    parser.add_argument("--adapter-gate-cap", type=float, default=0.25)
    parser.add_argument("--positive-patch-prob", type=float, default=0.35)
    parser.add_argument("--hard-negative-patch-prob", type=float, default=0.35)
    parser.add_argument("--negative-max-fg-fraction", type=float, default=0.001)
    parser.add_argument("--negative-sampling-after-steps", type=int, default=30)
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
    return args


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
