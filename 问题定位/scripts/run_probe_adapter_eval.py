import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
GT_DIR = ROOT / "datasets" / "ReXGroundingCT_subset" / "segmentations"
BASELINE_DIR = ROOT / "outputs" / "voxtell_val"
PROBE_CASES_PATH = ROOT / "问题定位" / "v4_probe_cases.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "probe_eval_metrics.json"


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


def load_probe_case_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ordered: list[str] = []
    seen: set[str] = set()
    for items in data.values():
        for item in items:
            case_name = item["case"]
            if case_name not in seen:
                seen.add(case_name)
                ordered.append(case_name)
    return ordered


def load_val_case_lookup() -> dict[str, dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    return {case["name"]: case for case in data["val"]}


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
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
    parser = argparse.ArgumentParser(description="Evaluate adapter predictions on the fixed probe set.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--probe-cases", type=Path, default=PROBE_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    case_lookup = load_val_case_lookup()
    probe_case_names = load_probe_case_names(args.probe_cases)
    label = args.label or args.adapter_dir.name

    results = {
        "label": label,
        "probe_cases": probe_case_names,
        "cases": [],
        "aggregate": {},
        "by_category": {},
        "by_lesion": {},
    }
    all_metrics = {"raw_baseline": [], "raw_adapter": []}
    by_category = defaultdict(lambda: {"raw_baseline": [], "raw_adapter": []})
    by_lesion = defaultdict(lambda: {"raw_baseline": [], "raw_adapter": []})

    for case_name in probe_case_names:
        case = case_lookup[case_name]
        baseline_summary_path = BASELINE_DIR / case_name.replace(".nii.gz", "") / "summary_comparison.json"
        adapter_summary_path = args.adapter_dir / case_name.replace(".nii.gz", "") / "summary_adapter_raw.json"
        if not baseline_summary_path.exists():
            raise FileNotFoundError(f"Missing baseline summary: {baseline_summary_path}")
        if not adapter_summary_path.exists():
            raise FileNotFoundError(f"Missing adapter summary: {adapter_summary_path}")

        gt = np.asanyarray(nib.load(str(GT_DIR / case_name)).dataobj)
        baseline_items = json.loads(baseline_summary_path.read_text(encoding="utf-8"))["comparison"]
        adapter_summary = json.loads(adapter_summary_path.read_text(encoding="utf-8"))
        adapter_items = adapter_summary["items"]
        if len(baseline_items) != len(adapter_items):
            raise ValueError(f"Item count mismatch for {case_name}")

        case_result = {"case": case_name, "mode": adapter_summary.get("mode"), "items": []}
        case_metrics = {"raw_baseline": [], "raw_adapter": []}
        for baseline_item, adapter_item in zip(baseline_items, adapter_items):
            idx = int(baseline_item["index"])
            gt_mask = gt[idx] > 0
            raw_pred = np.asanyarray(nib.load(baseline_item["raw_output"]).dataobj) > 0
            adapter_pred = np.asanyarray(nib.load(adapter_item["output"]).dataobj) > 0

            raw_metrics = compute_metrics(raw_pred, gt_mask)
            adapter_metrics = compute_metrics(adapter_pred, gt_mask)
            all_metrics["raw_baseline"].append(raw_metrics)
            all_metrics["raw_adapter"].append(adapter_metrics)
            case_metrics["raw_baseline"].append(raw_metrics)
            case_metrics["raw_adapter"].append(adapter_metrics)

            category_code = baseline_item.get("category") or case["categories"].get(str(idx))
            category_name = CATEGORY_LABELS.get(category_code, category_code or "unknown")
            lesion_name = (baseline_item.get("parsed_fields") or {}).get("lesion") or "unknown"
            by_category[category_name]["raw_baseline"].append(raw_metrics)
            by_category[category_name]["raw_adapter"].append(adapter_metrics)
            by_lesion[lesion_name]["raw_baseline"].append(raw_metrics)
            by_lesion[lesion_name]["raw_adapter"].append(adapter_metrics)

            case_result["items"].append(
                {
                    "index": idx,
                    "category_code": category_code,
                    "category_name": category_name,
                    "lesion_name": lesion_name,
                    "raw_prompt": baseline_item["raw_prompt"],
                    "suppression_mean": adapter_item.get("suppression_mean"),
                    "raw_baseline": raw_metrics,
                    "raw_adapter": adapter_metrics,
                    "delta_dice": adapter_metrics["dice"] - raw_metrics["dice"],
                    "delta_precision": adapter_metrics["precision"] - raw_metrics["precision"],
                    "delta_recall": adapter_metrics["recall"] - raw_metrics["recall"],
                }
            )

        case_result["aggregate"] = {
            "raw_baseline": summarize_metric_list(case_metrics["raw_baseline"]),
            "raw_adapter": summarize_metric_list(case_metrics["raw_adapter"]),
        }
        results["cases"].append(case_result)

    results["aggregate"] = {
        "raw_baseline": summarize_metric_list(all_metrics["raw_baseline"]),
        "raw_adapter": summarize_metric_list(all_metrics["raw_adapter"]),
    }
    for group_name, grouped_metrics in sorted(by_category.items()):
        results["by_category"][group_name] = {
            "raw_baseline": summarize_metric_list(grouped_metrics["raw_baseline"]),
            "raw_adapter": summarize_metric_list(grouped_metrics["raw_adapter"]),
        }
    for group_name, grouped_metrics in sorted(by_lesion.items()):
        results["by_lesion"][group_name] = {
            "raw_baseline": summarize_metric_list(grouped_metrics["raw_baseline"]),
            "raw_adapter": summarize_metric_list(grouped_metrics["raw_adapter"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved probe metrics to {args.output}")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
