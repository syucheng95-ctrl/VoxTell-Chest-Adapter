import json
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
GT_DIR = ROOT / "datasets" / "ReXGroundingCT_subset" / "segmentations"
PRED_DIR = ROOT / "outputs" / "voxtell_subset"
OUTPUT_PATH = ROOT / "outputs" / "voxtell_subset_metrics.json"

SELECTED_CASES = [
    "train_13249_b_2.nii.gz",
    "train_13631_a_1.nii.gz",
    "train_13577_a_2.nii.gz",
]


def load_case_metadata() -> dict[str, dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    rows = {}
    for split in ("train", "val", "test"):
        for item in data.get(split, []):
            if item["name"] in SELECTED_CASES:
                rows[item["name"]] = item
    return rows


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


def main() -> None:
    case_meta = load_case_metadata()
    results = {"cases": [], "aggregate": {}}
    all_metrics = {"raw": [], "structured": []}

    for case_name in SELECTED_CASES:
        meta = case_meta[case_name]
        gt = np.asanyarray(nib.load(str(GT_DIR / case_name)).dataobj)
        comparison = json.loads(
            (PRED_DIR / case_name.replace(".nii.gz", "") / "summary_comparison.json").read_text(encoding="utf-8")
        )["comparison"]

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

            case_result["items"].append(
                {
                    "index": key,
                    "raw_prompt": item["raw_prompt"],
                    "structured_prompt": item["structured_prompt"],
                    "raw": raw_metrics,
                    "structured": structured_metrics,
                }
            )
        results["cases"].append(case_result)

    for mode in ("raw", "structured"):
        metrics = all_metrics[mode]
        results["aggregate"][mode] = {
            "count": len(metrics),
            "mean_dice": float(np.mean([m["dice"] for m in metrics])),
            "mean_iou": float(np.mean([m["iou"] for m in metrics])),
            "mean_precision": float(np.mean([m["precision"] for m in metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in metrics])),
            "sum_tp": int(sum(m["tp"] for m in metrics)),
            "sum_fp": int(sum(m["fp"] for m in metrics)),
            "sum_fn": int(sum(m["fn"] for m in metrics)),
            "micro_dice": float(
                (2.0 * sum(m["tp"] for m in metrics))
                / max(1, (2 * sum(m["tp"] for m in metrics) + sum(m["fp"] for m in metrics) + sum(m["fn"] for m in metrics)))
            ),
            "micro_iou": float(
                sum(m["tp"] for m in metrics)
                / max(1, (sum(m["tp"] for m in metrics) + sum(m["fp"] for m in metrics) + sum(m["fn"] for m in metrics)))
            ),
        }

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {OUTPUT_PATH}")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
