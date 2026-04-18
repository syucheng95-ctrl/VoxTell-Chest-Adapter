import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DEFAULT = Path(r"C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe")
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_voxtell_adapter.py"
PROBE_INFER_SCRIPT = ROOT / "问题定位" / "scripts" / "ablate_v4_suppression.py"
PROBE_EVAL_SCRIPT = ROOT / "问题定位" / "scripts" / "run_probe_adapter_eval.py"
COMPARE_SCRIPT = ROOT / "问题定位" / "scripts" / "compare_probe_cases.py"
ANALYZE_SCRIPT = ROOT / "问题定位" / "scripts" / "analyze_v4_suppression_behavior.py"


def run_command(command: list[str], workdir: Path) -> None:
    print("Running:")
    print(" ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v5 pilot training matrix and probe evaluation.")
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "voxtell_adapter_v5_pilot_matrix")
    parser.add_argument("--probe-output-root", type=Path, default=ROOT / "outputs" / "probe_v5_pilot_matrix")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-findings-per-case", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--save-every-steps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.probe_output_root.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "v4_current",
            "extra_args": [
                "--suppression-target-mode", "binary",
                "--negative-fp-weight", "0.35",
            ],
        },
        {
            "name": "v5_soft_target",
            "extra_args": [
                "--suppression-target-mode", "soft_by_sample_mode",
                "--suppression-target-positive", "1.0",
                "--suppression-target-hard-negative", "0.4",
                "--suppression-target-random-negative", "0.0",
                "--negative-fp-weight", "0.35",
            ],
        },
        {
            "name": "v5_soft_target_fp",
            "extra_args": [
                "--suppression-target-mode", "soft_by_sample_mode",
                "--suppression-target-positive", "1.0",
                "--suppression-target-hard-negative", "0.4",
                "--suppression-target-random-negative", "0.0",
                "--negative-fp-weight", "0.40",
                "--negative-loss-scale", "2.75",
            ],
        },
    ]

    probe_metric_paths: list[Path] = []
    probe_labels: list[str] = []

    for variant in variants:
        run_dir = args.output_root / variant["name"]
        probe_root = args.probe_output_root / variant["name"]
        probe_metrics = ROOT / "outputs" / f"{variant['name']}_probe_metrics.json"
        probe_metric_paths.append(probe_metrics)
        probe_labels.append(variant["name"])

        train_cmd = [
            str(args.python),
            str(TRAIN_SCRIPT),
            "--output-dir", str(run_dir),
            "--epochs", str(args.epochs),
            "--max-cases", str(args.max_cases),
            "--max-findings-per-case", str(args.max_findings_per_case),
            "--max-steps", str(args.max_steps),
            "--save-every-steps", str(args.save_every_steps),
            "--skip-easy-empty-negatives",
        ] + variant["extra_args"]

        if args.force or not (run_dir / "train_metrics.json").exists():
            run_command(train_cmd, ROOT)
        else:
            print(f"Skipping training for {variant['name']} (existing output)")

        probe_cmd = [
            str(args.python),
            str(PROBE_INFER_SCRIPT),
            "--adapter-run-dir", str(run_dir),
            "--output-root", str(probe_root),
            "--mode", "full",
            "--force",
        ]
        if args.force or not (probe_root / "v4_full").exists():
            run_command(probe_cmd, ROOT)
        else:
            print(f"Skipping probe inference for {variant['name']} (existing output)")

        eval_cmd = [
            str(args.python),
            str(PROBE_EVAL_SCRIPT),
            "--adapter-dir", str(probe_root / "v4_full"),
            "--output", str(probe_metrics),
            "--label", variant["name"],
        ]
        run_command(eval_cmd, ROOT)

    compare_json = ROOT / "问题定位" / "v5_pilot_probe_comparison.json"
    compare_md = ROOT / "问题定位" / "v5_pilot_probe_comparison.md"
    compare_cmd = [
        str(args.python),
        str(COMPARE_SCRIPT),
        "--inputs",
        *[str(path) for path in probe_metric_paths],
        "--labels",
        *probe_labels,
        "--output-md", str(compare_md),
        "--output-json", str(compare_json),
    ]
    run_command(compare_cmd, ROOT)

    for variant in variants:
        analysis_json = ROOT / "问题定位" / f"{variant['name']}_suppression_analysis.json"
        analysis_md = ROOT / "问题定位" / f"{variant['name']}_suppression_analysis.md"
        analysis_cmd = [
            str(args.python),
            str(ANALYZE_SCRIPT),
            "--train-metrics", str(args.output_root / variant["name"] / "train_metrics.json"),
            "--probe-metrics", str(ROOT / "outputs" / f"{variant['name']}_probe_metrics.json"),
            "--output-md", str(analysis_md),
            "--output-json", str(analysis_json),
        ]
        run_command(analysis_cmd, ROOT)

    summary = {
        "variants": variants,
        "probe_metrics": [str(path) for path in probe_metric_paths],
        "comparison_md": str(compare_md),
        "comparison_json": str(compare_json),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
