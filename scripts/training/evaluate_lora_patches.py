import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydoc
import torch
from peft import PeftModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
VOXTELL_ROOT = ROOT / "VoxTell"
UTILS_ROOT = ROOT / "scripts" / "utils"
for extra_path in [str(VOXTELL_ROOT), str(UTILS_ROOT)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from voxtell.model.voxtell_model import VoxTellModel

from train_voxtell_lora import (
    DATASET_JSON,
    MODEL_DIR,
    PATCH_SIZE,
    TRAIN_DATASET_DIR,
    build_network,
    compute_patch_center,
    crop_mask_by_bbox,
    dice_loss_from_logits,
    extract_patch,
    flatten_samples,
    get_or_create_text_cache,
    load_subset_metadata,
    preprocess_image,
)


def compute_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict:
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.float()
    intersection = float((pred * target).sum().cpu())
    pred_sum = float(pred.sum().cpu())
    target_sum = float(target.sum().cpu())
    union = pred_sum + target_sum - intersection
    dice = (2 * intersection / (pred_sum + target_sum)) if (pred_sum + target_sum) > 0 else 1.0
    iou = (intersection / union) if union > 0 else 1.0
    precision = (intersection / pred_sum) if pred_sum > 0 else 0.0
    recall = (intersection / target_sum) if target_sum > 0 else 0.0
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA VoxTell on lesion-centered train patches.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, default=ROOT / "hf_cache")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-findings-per-case", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_CACHE"] = str(args.hf_home / "hub")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = load_subset_metadata()[: args.max_cases]
    samples = flatten_samples(cases)
    text_cache = get_or_create_text_cache(args.adapter_dir.parent, [sample.prompt for sample in samples])
    normalization = ZScoreNormalization(intensityproperties={})

    base_model = build_network(device).eval()
    lora_model = build_network(device)
    lora_model = PeftModel.from_pretrained(lora_model, str(args.adapter_dir), is_trainable=False).eval()

    rows = []
    for case in tqdm(cases, desc="Evaluating"):
        image_arr, seg_arr = load_case_arrays(case["name"], normalization)
        finding_keys = sorted(map(int, case["findings"].keys()))[: args.max_findings_per_case]
        for finding_idx in finding_keys:
            prompt = case["findings"][str(finding_idx)]
            category_code = case.get("categories", {}).get(str(finding_idx))
            mask_arr = seg_arr[finding_idx]
            center = compute_patch_center(mask_arr)
            image_patch = extract_patch(image_arr, center, PATCH_SIZE, fill_value=0.0)
            mask_patch = extract_patch(mask_arr, center, PATCH_SIZE, fill_value=0.0)
            image_tensor = torch.from_numpy(image_patch[None]).to(device=device, dtype=torch.float32)
            target_tensor = torch.from_numpy(mask_patch[None, None]).to(device=device, dtype=torch.float32)
            text_embedding = text_cache[prompt].view(1, 1, -1).to(device=device, dtype=torch.float32)

            with torch.inference_mode():
                base_logits = base_model(image_tensor, text_embedding)
                lora_logits = lora_model(image_tensor, text_embedding)

            rows.append(
                {
                    "case_name": case["name"],
                    "finding_idx": finding_idx,
                    "category_code": category_code,
                    "prompt": prompt,
                    "base": compute_metrics(base_logits, target_tensor),
                    "lora": compute_metrics(lora_logits, target_tensor),
                }
            )

    def mean_metric(key: str, mode: str) -> float:
        return float(np.mean([row[mode][key] for row in rows])) if rows else 0.0

    payload = {
        "adapter_dir": str(args.adapter_dir),
        "cases_evaluated": len(cases),
        "findings_evaluated": len(rows),
        "summary": {
            "base": {k: mean_metric(k, "base") for k in ["dice", "iou", "precision", "recall"]},
            "lora": {k: mean_metric(k, "lora") for k in ["dice", "iou", "precision", "recall"]},
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
