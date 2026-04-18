import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V3_METRICS = ROOT / "outputs" / "voxtell_val_adapter_raw_v3_metrics.json"
V4_METRICS = ROOT / "outputs" / "voxtell_val_adapter_raw_v4_metrics.json"
OUTPUT_JSON = ROOT / "问题定位" / "v4_probe_cases.json"
OUTPUT_MD = ROOT / "问题定位" / "v4_probe_cases.md"

SMALL_TARGET_MAX_GT_VOXELS = 8_000
DIFFUSE_MIN_GT_VOXELS = 150_000


@dataclass
class CaseSummary:
    case: str
    v3_mean_dice: float
    v4_mean_dice: float
    delta_mean_dice: float
    gt_voxels: int
    pred_voxels_v4: int
    item_count: int
    categories: list[str]
    lesions: list[str]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_case_summaries(v3_data: dict, v4_data: dict) -> list[CaseSummary]:
    summaries: list[CaseSummary] = []
    for case_v3, case_v4 in zip(v3_data["cases"], v4_data["cases"]):
        items_v3 = case_v3["items"]
        items_v4 = case_v4["items"]
        v3_mean_dice = sum(item["raw_adapter"]["dice"] for item in items_v3) / len(items_v3)
        v4_mean_dice = sum(item["raw_adapter"]["dice"] for item in items_v4) / len(items_v4)
        summaries.append(
            CaseSummary(
                case=case_v3["case"],
                v3_mean_dice=v3_mean_dice,
                v4_mean_dice=v4_mean_dice,
                delta_mean_dice=v4_mean_dice - v3_mean_dice,
                gt_voxels=sum(item["raw_adapter"]["gt_voxels"] for item in items_v4),
                pred_voxels_v4=sum(item["raw_adapter"]["pred_voxels"] for item in items_v4),
                item_count=len(items_v4),
                categories=sorted({item["category_name"] for item in items_v4}),
                lesions=sorted({item["lesion_name"] for item in items_v4}),
            )
        )
    return summaries


def is_small_target(case: CaseSummary) -> bool:
    return case.gt_voxels <= SMALL_TARGET_MAX_GT_VOXELS


def is_diffuse_target(case: CaseSummary) -> bool:
    diffuse_categories = {
        "ground_glass_opacity",
        "consolidation",
        "emphysema",
        "infiltration_or_interstitial_opacity",
    }
    return case.gt_voxels >= DIFFUSE_MIN_GT_VOXELS or any(cat in diffuse_categories for cat in case.categories)


def choose_case(candidates: list[CaseSummary], used: set[str]) -> CaseSummary | None:
    for case in candidates:
        if case.case not in used:
            used.add(case.case)
            return case
    return None


def select_probe_cases(summaries: list[CaseSummary]) -> dict:
    used: set[str] = set()
    sorted_by_delta = sorted(summaries, key=lambda case: case.delta_mean_dice)
    sorted_small = sorted(
        [case for case in summaries if is_small_target(case)],
        key=lambda case: (case.gt_voxels, -abs(case.delta_mean_dice)),
    )
    sorted_diffuse = sorted(
        [case for case in summaries if is_diffuse_target(case)],
        key=lambda case: (-case.gt_voxels, case.delta_mean_dice),
    )

    selected = {
        "v4_worse_than_v3": [],
        "v4_better_than_v3": [],
        "small_target_cases": [],
        "diffuse_cases": [],
    }

    for case in sorted_by_delta[:8]:
        if len(selected["v4_worse_than_v3"]) >= 2:
            break
        if picked := choose_case([case], used):
            selected["v4_worse_than_v3"].append(picked)

    for case in reversed(sorted_by_delta[-8:]):
        if len(selected["v4_better_than_v3"]) >= 2:
            break
        if picked := choose_case([case], used):
            selected["v4_better_than_v3"].append(picked)

    for case in sorted_small:
        if len(selected["small_target_cases"]) >= 2:
            break
        if picked := choose_case([case], used):
            selected["small_target_cases"].append(picked)

    for case in sorted_diffuse:
        if len(selected["diffuse_cases"]) >= 2:
            break
        if picked := choose_case([case], used):
            selected["diffuse_cases"].append(picked)

    return selected


def to_serializable(selected: dict) -> dict:
    return {
        group: [
            {
                "case": item.case,
                "v3_mean_dice": item.v3_mean_dice,
                "v4_mean_dice": item.v4_mean_dice,
                "delta_mean_dice": item.delta_mean_dice,
                "gt_voxels": item.gt_voxels,
                "pred_voxels_v4": item.pred_voxels_v4,
                "item_count": item.item_count,
                "categories": item.categories,
                "lesions": item.lesions,
            }
            for item in items
        ]
        for group, items in selected.items()
    }


def render_markdown(selected: dict) -> str:
    lines = ["# v4 问题定位集候选", "", "以下 8 个 case 用于后续 suppression / category-aware 的问题定位。", ""]
    for group, items in selected.items():
        lines.append(f"## {group}")
        lines.append("")
        for item in items:
            lines.append(f"- `{item['case']}`")
            lines.append(f"  `delta_mean_dice = {item['delta_mean_dice']:.4f}`, `v3 = {item['v3_mean_dice']:.4f}`, `v4 = {item['v4_mean_dice']:.4f}`")
            lines.append(f"  `gt_voxels = {item['gt_voxels']}`, `pred_voxels_v4 = {item['pred_voxels_v4']}`, `item_count = {item['item_count']}`")
            lines.append(f"  `categories = {', '.join(item['categories'])}`")
            lines.append(f"  `lesions = {', '.join(item['lesions'])}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    v3_data = load_metrics(V3_METRICS)
    v4_data = load_metrics(V4_METRICS)
    summaries = build_case_summaries(v3_data, v4_data)
    selected = select_probe_cases(summaries)
    serializable = to_serializable(selected)
    OUTPUT_JSON.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(serializable), encoding="utf-8")
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_MD}")
    print(json.dumps(serializable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
