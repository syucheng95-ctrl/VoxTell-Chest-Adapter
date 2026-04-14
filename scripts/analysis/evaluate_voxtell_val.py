import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
GT_DIR = ROOT / "datasets" / "ReXGroundingCT_subset" / "segmentations"
PRED_DIR = ROOT / "outputs" / "voxtell_val"
OUTPUT_PATH = ROOT / "outputs" / "voxtell_val_metrics.json"


CATEGORY_LABELS = {
    "1a": "bronchial_wall_thickening",
    "1b": "bronchiectasis",
    "1c": "emphysema",
    "1d": "fibrosis_or_scarring",
    "1e": "infiltration_or_interstitial_opacity",
    "1f": "pleural_effusion_or_thickening",
    "2a": "atelectasis",
    "2b": "consolidation",
    "2c": "ground_glass_opacity",
    "2d": "nodule",
    "2e": "mass",
    "2h": "other_focal_lesion",
}


def load_val_cases(limit: int | None) -> list[dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    cases = list(data["val"])
    return cases[:limit] if limit else cases


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    dice = 1.0 if pred_sum == 0 and gt_sum == 0 else (2.0 * tp) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else 0.0
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if pred_sum == 0 and gt_sum == 0 else tp / pred_sum if pred_sum else 0.0
    recall = 1.0 if gt_sum == 0 else tp / gt_sum

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_voxels": pred_sum,
        "gt_voxels": gt_sum,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def summarize_metric_list(metrics: list[dict]) -> dict[str, float | int]:
    tp = sum(m["tp"] for m in metrics)
    fp = sum(m["fp"] for m in metrics)
    fn = sum(m["fn"] for m in metrics)
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    return {
        "count": len(metrics),
        "mean_dice": float(np.mean([m["dice"] for m in metrics])) if metrics else 0.0,
        "mean_iou": float(np.mean([m["iou"] for m in metrics])) if metrics else 0.0,
        "mean_precision": float(np.mean([m["precision"] for m in metrics])) if metrics else 0.0,
        "mean_recall": float(np.mean([m["recall"] for m in metrics])) if metrics else 0.0,
        "sum_tp": int(tp),
        "sum_fp": int(fp),
        "sum_fn": int(fn),
        "micro_dice": float((2.0 * tp) / denom_dice) if denom_dice else 0.0,
        "micro_iou": float(tp / denom_iou) if denom_iou else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VoxTell raw vs structured prompts on the ReXGroundingCT val set.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N val cases.")
    args = parser.parse_args()

    cases = load_val_cases(args.limit)
    results = {"cases": [], "aggregate": {}, "by_category": {}, "by_lesion": {}}

    all_metrics = {"raw": [], "structured": []}
    by_category = defaultdict(lambda: {"raw": [], "structured": []})
    by_lesion = defaultdict(lambda: {"raw": [], "structured": []})

    for case in cases:
        case_name = case["name"]
        summary_path = PRED_DIR / case_name.replace(".nii.gz", "") / "summary_comparison.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing prediction summary: {summary_path}")

        gt = np.asanyarray(nib.load(str(GT_DIR / case_name)).dataobj)
        comparison = json.loads(summary_path.read_text(encoding="utf-8"))["comparison"]
        case_result = {"case": case_name, "items": []}

        for item in comparison:
            key = int(item["index"])
            gt_mask = gt[key] > 0
            raw_pred = np.asanyarray(nib.load(item["raw_output"]).dataobj) > 0
            structured_pred = np.asanyarray(nib.load(item["structured_output"]).dataobj) > 0

            raw_metrics = compute_metrics(raw_pred, gt_mask)
            structured_metrics = compute_metrics(structured_pred, gt_mask)
            all_metrics["raw"].append(raw_metrics)
            all_metrics["structured"].append(structured_metrics)

            category_code = item.get("category") or case["categories"].get(str(key))
            category_name = CATEGORY_LABELS.get(category_code, category_code or "unknown")
            lesion_name = (item.get("parsed_fields") or {}).get("lesion") or "unknown"

            by_category[category_name]["raw"].append(raw_metrics)
            by_category[category_name]["structured"].append(structured_metrics)
            by_lesion[lesion_name]["raw"].append(raw_metrics)
            by_lesion[lesion_name]["structured"].append(structured_metrics)

            case_result["items"].append(
                {
                    "index": key,
                    "category_code": category_code,
                    "category_name": category_name,
                    "lesion_name": lesion_name,
                    "raw_prompt": item["raw_prompt"],
                    "structured_prompt": item["structured_prompt"],
                    "raw": raw_metrics,
                    "structured": structured_metrics,
                }
            )
        results["cases"].append(case_result)

    for mode in ("raw", "structured"):
        results["aggregate"][mode] = summarize_metric_list(all_metrics[mode])

    for group_name, grouped_metrics in sorted(by_category.items()):
        results["by_category"][group_name] = {
            "raw": summarize_metric_list(grouped_metrics["raw"]),
            "structured": summarize_metric_list(grouped_metrics["structured"]),
        }

    for group_name, grouped_metrics in sorted(by_lesion.items()):
        results["by_lesion"][group_name] = {
            "raw": summarize_metric_list(grouped_metrics["raw"]),
            "structured": summarize_metric_list(grouped_metrics["structured"]),
        }

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {OUTPUT_PATH}")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
