from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "voxtell_v1.1" / "fold_0" / "checkpoint_final.pth"
OUTPUT_DIR = ROOT / "models" / "voxtell_v1.1" / "split_modules"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["network_weights"]

    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in state_dict.items():
        prefix = key.split(".")[0]
        groups[prefix][key] = value

    summary_lines = [
        f"source_checkpoint: {CHECKPOINT_PATH}",
        f"total_tensors: {len(state_dict)}",
        "",
        "saved_modules:",
    ]

    for module_name in sorted(groups):
        module_state = groups[module_name]
        out_path = OUTPUT_DIR / f"{module_name}.pth"
        torch.save(
            {
                "module_name": module_name,
                "source_checkpoint": str(CHECKPOINT_PATH),
                "num_tensors": len(module_state),
                "state_dict": module_state,
            },
            out_path,
        )
        summary_lines.append(f"- {module_name}: {len(module_state)} tensors -> {out_path.name}")

    # Common combinations that are likely useful for transfer or partial loading.
    combos = {
        "encoder_only.pth": ["encoder"],
        "decoder_only.pth": ["decoder"],
        "vision_backbone.pth": ["encoder", "decoder"],
        "prompt_fusion_only.pth": [
            "transformer_decoder",
            "project_text_embed",
            "project_bottleneck_embed",
            "project_to_decoder_channels",
            "pos_embed",
        ],
        "full_network_weights_only.pth": sorted(groups.keys()),
    }

    summary_lines.extend(["", "saved_combinations:"])
    for filename, prefixes in combos.items():
        combo_state = {}
        for prefix in prefixes:
            combo_state.update(groups[prefix])
        out_path = OUTPUT_DIR / filename
        torch.save(
            {
                "module_names": prefixes,
                "source_checkpoint": str(CHECKPOINT_PATH),
                "num_tensors": len(combo_state),
                "state_dict": combo_state,
            },
            out_path,
        )
        summary_lines.append(f"- {filename}: {prefixes} ({len(combo_state)} tensors)")

    (OUTPUT_DIR / "README.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Split checkpoint written to: {OUTPUT_DIR}")
    for line in summary_lines[4:]:
        print(line)


if __name__ == "__main__":
    main()
