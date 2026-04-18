import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_METRICS = ROOT / "outputs" / "voxtell_adapter_v4_fullrun" / "train_metrics.json"
DEFAULT_OUTPUT_MD = ROOT / "问题定位" / "v4_suppression_behavior.md"
DEFAULT_OUTPUT_JSON = ROOT / "问题定位" / "v4_suppression_behavior.json"


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_entries(entries: list[dict]) -> dict:
    return {
        "count": len(entries),
        "mean_loss": mean([float(entry["loss"]) for entry in entries]),
        "mean_seg_loss": mean([float(entry["seg_loss"]) for entry in entries]),
        "mean_fp_penalty": mean([float(entry["fp_penalty"]) for entry in entries]),
        "mean_patch_fg_fraction": mean([float(entry["patch_fg_fraction"]) for entry in entries]),
        "mean_suppression_mean": mean([float(entry["suppression_mean"]) for entry in entries if entry.get("suppression_mean") is not None]),
        "mean_suppression_loss": mean([float(entry["suppression_loss"]) for entry in entries if entry.get("suppression_loss") is not None]),
        "mean_bce": mean([float(entry["bce"]) for entry in entries]),
        "mean_dice_loss": mean([float(entry["dice_loss"]) for entry in entries]),
    }


def summarize_probe_metrics(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    item_rows = [item for case in data["cases"] for item in case["items"]]

    def with_suppression() -> list[dict]:
        return [item for item in item_rows if item.get("suppression_mean") is not None]

    return {
        "label": data.get("label"),
        "aggregate": data["aggregate"]["raw_adapter"],
        "suppression_mean_over_items": mean([float(item["suppression_mean"]) for item in with_suppression()]),
        "suppression_mean_when_fp_gt_0": mean(
            [float(item["suppression_mean"]) for item in with_suppression() if item["raw_adapter"]["fp"] > 0]
        ),
        "suppression_mean_when_dice_gt_0": mean(
            [float(item["suppression_mean"]) for item in with_suppression() if item["raw_adapter"]["dice"] > 0]
        ),
        "mean_fp_over_items": mean([float(item["raw_adapter"]["fp"]) for item in item_rows]),
        "mean_pred_voxels_over_items": mean([float(item["raw_adapter"]["pred_voxels"]) for item in item_rows]),
    }


def render_markdown(analysis: dict) -> str:
    lines = ["# v4 Suppression Behavior Analysis", ""]
    lines.append("## Training Diagnostics")
    lines.append("")
    diag = analysis["train_diagnostics"]
    for key, value in diag.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## By Sample Mode")
    lines.append("")
    for mode, summary in analysis["by_sample_mode"].items():
        lines.append(f"- `{mode}`")
        for key, value in summary.items():
            lines.append(f"  `{key}`: `{value}`")
    lines.append("")

    if analysis.get("probe_metrics") is not None:
        lines.append("## Probe Summary")
        lines.append("")
        probe = analysis["probe_metrics"]
        lines.append(f"- `label`: `{probe['label']}`")
        for key, value in probe["aggregate"].items():
            lines.append(f"- `aggregate.{key}`: `{value}`")
        lines.append(f"- `suppression_mean_over_items`: `{probe['suppression_mean_over_items']}`")
        lines.append(f"- `suppression_mean_when_fp_gt_0`: `{probe['suppression_mean_when_fp_gt_0']}`")
        lines.append(f"- `suppression_mean_when_dice_gt_0`: `{probe['suppression_mean_when_dice_gt_0']}`")
        lines.append(f"- `mean_fp_over_items`: `{probe['mean_fp_over_items']}`")
        lines.append(f"- `mean_pred_voxels_over_items`: `{probe['mean_pred_voxels_over_items']}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v4 suppression behavior from training and probe metrics.")
    parser.add_argument("--train-metrics", type=Path, default=DEFAULT_TRAIN_METRICS)
    parser.add_argument("--probe-metrics", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    train_data = json.loads(args.train_metrics.read_text(encoding="utf-8"))
    loss_history = train_data["loss_history"]
    by_mode = {}
    for mode in sorted({entry["sample_mode"] for entry in loss_history}):
        by_mode[mode] = summarize_entries([entry for entry in loss_history if entry["sample_mode"] == mode])

    analysis = {
        "train_metrics_path": str(args.train_metrics),
        "train_diagnostics": train_data.get("diagnostics", {}),
        "by_sample_mode": by_mode,
        "probe_metrics": summarize_probe_metrics(args.probe_metrics),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
