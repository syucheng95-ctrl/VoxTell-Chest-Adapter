import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
VOXTELL_ROOT = ROOT / "VoxTell"
UTILS_ROOT = ROOT / "scripts" / "utils"
for extra_path in [str(VOXTELL_ROOT), str(UTILS_ROOT)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from rex_prompt_tools import parse_finding
from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_subset"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voxtell_val_lora_raw"


def embed_text_on_cpu(predictor: VoxTellPredictor, text_prompts: list[str], target_device: torch.device) -> torch.Tensor:
    prompts = wrap_with_instruction(text_prompts)
    tokens = predictor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=predictor.max_text_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        predictor.text_backbone = predictor.text_backbone.to("cpu")
        outputs = predictor.text_backbone(**tokens)
        embeddings = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
    return embeddings.view(1, len(text_prompts), -1).to(target_device)


def load_val_cases(limit: int | None) -> list[dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    cases = list(data["val"])
    return cases[:limit] if limit else cases


def slugify(prompt: str) -> str:
    prompt_slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    return "_".join(filter(None, prompt_slug.split("_")))[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoxTell raw prompts on full val using a trained LoRA adapter.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Adapter dir: {args.adapter_dir}")
    print(f"Output dir: {args.output_dir}")

    predictor = VoxTellPredictor(model_dir=str(MODEL_DIR), device=device)
    predictor.network = PeftModel.from_pretrained(predictor.network, str(args.adapter_dir), is_trainable=False).to(device)
    predictor.network.eval()
    io = NibabelIOWithReorient()

    cases = load_val_cases(args.limit)
    print(f"Loaded {len(cases)} val cases")

    for idx, case in enumerate(cases, start=1):
        case_name = case["name"]
        case_output_dir = args.output_dir / case_name.replace(".nii.gz", "")
        summary_path = case_output_dir / "summary_lora_raw.json"
        if summary_path.exists() and not args.force:
            print(f"[{idx}/{len(cases)}] Skipping {case_name} (already completed)")
            continue

        print(f"[{idx}/{len(cases)}] Processing {case_name}")
        image_path = DATASET_DIR / "images_flat" / case_name
        raw_prompts = [case["findings"][str(i)] for i in sorted(map(int, case["findings"].keys()))]
        parsed_prompts = [parse_finding(prompt) for prompt in raw_prompts]

        image, properties = io.read_images([str(image_path)])
        image = np.asarray(image)
        preprocessed, bbox, original_shape = predictor.preprocess(image)

        embeddings = embed_text_on_cpu(predictor, raw_prompts, device)
        logits = predictor.predict_sliding_window_return_logits(preprocessed, embeddings).to("cpu")
        with torch.no_grad():
            prediction = (torch.sigmoid(logits.float()) > 0.5).numpy().astype(np.uint8)

        restored = np.zeros((prediction.shape[0], *original_shape), dtype=np.uint8)
        x_slice, y_slice, z_slice = bbox
        restored[:, x_slice[0]:x_slice[1], y_slice[0]:y_slice[1], z_slice[0]:z_slice[1]] = prediction

        raw_output_dir = case_output_dir / "raw_lora"
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for prompt_idx, prompt in enumerate(raw_prompts):
            output_path = raw_output_dir / f"{prompt_idx:02d}_{slugify(prompt)}.nii.gz"
            io.write_seg(restored[prompt_idx], str(output_path), properties)
            voxels = int(restored[prompt_idx].sum())
            rows.append(
                {
                    "index": prompt_idx,
                    "category": case["categories"].get(str(prompt_idx)),
                    "raw_prompt": prompt,
                    "parsed_fields": parsed_prompts[prompt_idx].to_dict(),
                    "voxels": voxels,
                    "output": str(output_path),
                }
            )
            print(f"    saved {output_path.name} voxels={voxels}")

        summary_path.write_text(
            json.dumps({"case": case_name, "items": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
