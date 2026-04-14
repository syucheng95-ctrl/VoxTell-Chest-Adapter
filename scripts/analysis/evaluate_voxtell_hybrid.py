import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "outputs" / "voxtell_val_metrics.json"
OUTPUT_PATH = ROOT / "outputs" / "voxtell_val_hybrid_metrics.json"


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
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    category_policy = {}
    for category_name, group in data["by_category"].items():
        raw = group["raw"]
        structured = group["structured"]
        if structured["mean_dice"] > raw["mean_dice"]:
            category_policy[category_name] = "structured"
        else:
            category_policy[category_name] = "raw"

    hybrid_metrics = []
    hybrid_cases = []
    mode_usage = {"raw": 0, "structured": 0}

    for case in data["cases"]:
        case_result = {"case": case["case"], "items": []}
        for item in case["items"]:
            category_name = item["category_name"]
            chosen_mode = category_policy[category_name]
            chosen_metrics = item[chosen_mode]
            hybrid_metrics.append(chosen_metrics)
            mode_usage[chosen_mode] += 1
            case_result["items"].append(
                {
                    "index": item["index"],
                    "category_name": category_name,
                    "lesion_name": item["lesion_name"],
                    "raw_prompt": item["raw_prompt"],
                    "chosen_mode": chosen_mode,
                    "raw": item["raw"],
                    "structured": item["structured"],
                    "hybrid": chosen_metrics,
                }
            )
        hybrid_cases.append(case_result)

    hybrid_summary = summarize_metric_list(hybrid_metrics)
    result = {
        "category_policy": category_policy,
        "mode_usage": mode_usage,
        "hybrid": hybrid_summary,
        "raw": data["aggregate"]["raw"],
        "structured": data["aggregate"]["structured"],
        "cases": hybrid_cases,
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
