import argparse
import json
from pathlib import Path

from rex_prompt_tools import format_structured_prompt, parse_finding


ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"


def load_findings(split: str, limit: int, contains: str | None) -> list[tuple[str, str]]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    results = []
    for item in data[split]:
        for finding in item["findings"].values():
            if contains and contains.lower() not in finding.lower():
                continue
            results.append((item["name"], finding))
            if len(results) >= limit:
                return results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect structured prompt parsing on ReXGroundingCT findings.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--contains", type=str, default=None, help="Only inspect findings containing this substring.")
    parser.add_argument("--style", choices=["templated", "compact"], default="templated")
    args = parser.parse_args()

    findings = load_findings(args.split, args.limit, args.contains)
    for idx, (case_name, finding) in enumerate(findings, start=1):
        parsed = parse_finding(finding)
        print(f"===== Sample {idx} =====")
        print(f"Case: {case_name}")
        print(f"Raw: {finding}")
        print(f"Parsed: {json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2)}")
        print(f"Structured Prompt: {format_structured_prompt(parsed, style=args.style)}")
        print()


if __name__ == "__main__":
    main()
