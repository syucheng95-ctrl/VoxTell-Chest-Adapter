import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_MD = ROOT / "问题定位" / "probe_case_comparison.md"
DEFAULT_OUTPUT_JSON = ROOT / "问题定位" / "probe_case_comparison.json"


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_snapshot(data: dict) -> dict:
    aggregate = data["aggregate"]["raw_adapter"]
    return {
        "mean_dice": aggregate["mean_dice"],
        "micro_dice": aggregate["micro_dice"],
        "mean_precision": aggregate["mean_precision"],
        "mean_recall": aggregate["mean_recall"],
        "count": aggregate["count"],
    }


def build_case_table(named_results: dict[str, dict]) -> dict[str, dict]:
    case_table: dict[str, dict] = {}
    for label, data in named_results.items():
        for case in data["cases"]:
            case_name = case["case"]
            case_table.setdefault(case_name, {})[label] = {
                "mean_dice": case["aggregate"]["raw_adapter"]["mean_dice"],
                "micro_dice": case["aggregate"]["raw_adapter"]["micro_dice"],
                "mean_precision": case["aggregate"]["raw_adapter"]["mean_precision"],
                "mean_recall": case["aggregate"]["raw_adapter"]["mean_recall"],
                "mode": case.get("mode"),
            }
    return case_table


def build_item_table(named_results: dict[str, dict]) -> dict[str, dict]:
    item_table: dict[str, dict] = {}
    for label, data in named_results.items():
        for case in data["cases"]:
            for item in case["items"]:
                key = f"{case['case']}::{item['index']}"
                item_table.setdefault(
                    key,
                    {
                        "case": case["case"],
                        "index": item["index"],
                        "category_name": item["category_name"],
                        "lesion_name": item["lesion_name"],
                        "raw_prompt": item["raw_prompt"],
                    },
                )[label] = {
                    "dice": item["raw_adapter"]["dice"],
                    "precision": item["raw_adapter"]["precision"],
                    "recall": item["raw_adapter"]["recall"],
                    "pred_voxels": item["raw_adapter"]["pred_voxels"],
                    "gt_voxels": item["raw_adapter"]["gt_voxels"],
                    "suppression_mean": item.get("suppression_mean"),
                }
    return item_table


def render_markdown(named_results: dict[str, dict], comparison: dict) -> str:
    lines = ["# Probe Case Comparison", ""]
    lines.append("## Aggregate")
    lines.append("")
    for label, snapshot in comparison["aggregate"].items():
        lines.append(
            f"- `{label}`: "
            f"`mean_dice={snapshot['mean_dice']:.4f}`, "
            f"`micro_dice={snapshot['micro_dice']:.4f}`, "
            f"`mean_precision={snapshot['mean_precision']:.4f}`, "
            f"`mean_recall={snapshot['mean_recall']:.4f}`, "
            f"`count={snapshot['count']}`"
        )
    lines.append("")

    lines.append("## Case Level")
    lines.append("")
    for case_name, case_values in sorted(comparison["cases"].items()):
        lines.append(f"- `{case_name}`")
        for label in named_results:
            if label not in case_values:
                continue
            values = case_values[label]
            lines.append(
                f"  `{label}`: "
                f"`mean_dice={values['mean_dice']:.4f}`, "
                f"`micro_dice={values['micro_dice']:.4f}`, "
                f"`precision={values['mean_precision']:.4f}`, "
                f"`recall={values['mean_recall']:.4f}`"
            )
    lines.append("")

    lines.append("## Largest Item Deltas")
    lines.append("")
    item_rows = []
    labels = list(named_results.keys())
    if len(labels) >= 2:
        left, right = labels[0], labels[1]
        for key, row in comparison["items"].items():
            if left not in row or right not in row:
                continue
            delta = row[right]["dice"] - row[left]["dice"]
            item_rows.append((delta, row))
        item_rows.sort(key=lambda x: x[0])
        for delta, row in item_rows[:5] + item_rows[-5:]:
            lines.append(
                f"- `{row['case']}` idx={row['index']} "
                f"`{row['category_name']}` "
                f"`delta_dice={delta:.4f}` "
                f"`prompt={row['raw_prompt']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple probe-eval metric files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    named_results = {label: load_metrics(path) for label, path in zip(args.labels, args.inputs)}
    comparison = {
        "aggregate": {label: aggregate_snapshot(data) for label, data in named_results.items()},
        "cases": build_case_table(named_results),
        "items": build_item_table(named_results),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(named_results, comparison), encoding="utf-8")
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
