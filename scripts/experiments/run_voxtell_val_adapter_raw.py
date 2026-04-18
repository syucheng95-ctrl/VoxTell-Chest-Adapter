import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

ROOT = Path(__file__).resolve().parents[2]
VOXTELL_ROOT = ROOT / "VoxTell"
UTILS_ROOT = ROOT / "scripts" / "utils"
TRAINING_ROOT = ROOT / "scripts" / "training"
for extra_path in [str(ROOT), str(VOXTELL_ROOT), str(UTILS_ROOT), str(TRAINING_ROOT)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "hf_cache" / "hub"))

from rex_prompt_tools import parse_finding
from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.text_embedding import (
    build_text_representations,
    collate_text_sequences,
    wrap_with_instruction,
)

from train_voxtell_adapter import ADAPTER_STATE_NAME, build_adapter_network


DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_subset"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voxtell_val_adapter_raw"


def embed_text_on_cpu(
    predictor: VoxTellPredictor,
    text_prompts: list[str],
    target_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pooled, token_sequences, token_masks = build_text_representations(
            outputs.last_hidden_state,
            tokens["attention_mask"],
        )
    token_embeddings, attention_mask = collate_text_sequences(
        token_sequences,
        token_masks,
        device=target_device,
        dtype=torch.float32,
    )
    return pooled.view(1, len(text_prompts), -1).to(target_device), token_embeddings, attention_mask


def load_val_cases(limit: int | None) -> list[dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    cases = list(data["val"])
    return cases[:limit] if limit else cases


def slugify(prompt: str) -> str:
    prompt_slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    return "_".join(filter(None, prompt_slug.split("_")))[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoxTell raw prompts on full val using a trained text-guided adapter.")
    parser.add_argument("--adapter-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-version", choices=["v6_4", "v6_5"], default="v6_4")
    parser.add_argument("--adapter-insertion-point", choices=["pre_decoder", "post_decoder"], default="pre_decoder")
    parser.add_argument("--adapter-num-groups", type=int, default=8)
    parser.add_argument("--adapter-candidate-scale", type=float, default=0.1)
    parser.add_argument("--adapter-risk-scale", type=float, default=0.08)
    parser.add_argument("--adapter-gate-cap", type=float, default=0.25)
    parser.add_argument("--adapter-category-scale", type=float, default=0.2)
    parser.add_argument("--adapter-refine-scale", type=float, default=0.035)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Adapter run dir: {args.adapter_run_dir}")
    print(f"Output dir: {args.output_dir}")

    adapter_state_path = args.adapter_run_dir / ADAPTER_STATE_NAME
    if not adapter_state_path.exists():
        raise FileNotFoundError(f"Missing adapter state: {adapter_state_path}")
    state_dict = torch.load(adapter_state_path, map_location=device, weights_only=False)
    predictor = VoxTellPredictor(model_dir=str(MODEL_DIR), device=device)
    predictor.network = build_adapter_network(
        device=device,
        adapter_version=args.adapter_version,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_insertion_point=args.adapter_insertion_point,
        adapter_num_groups=args.adapter_num_groups,
        adapter_risk_groups=None,
        adapter_candidate_scale=args.adapter_candidate_scale,
        adapter_risk_scale=args.adapter_risk_scale,
        adapter_gate_cap=args.adapter_gate_cap,
        adapter_category_scale=args.adapter_category_scale,
        adapter_refine_scale=args.adapter_refine_scale,
    )
    predictor.network.load_state_dict(state_dict, strict=False)
    predictor.network.eval()
    io = NibabelIOWithReorient()

    cases = load_val_cases(args.limit)
    print(f"Loaded {len(cases)} val cases")

    for idx, case in enumerate(cases, start=1):
        case_name = case["name"]
        case_output_dir = args.output_dir / case_name.replace(".nii.gz", "")
        summary_path = case_output_dir / "summary_adapter_raw.json"
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

        embeddings, text_token_embeddings, text_attention_mask = embed_text_on_cpu(predictor, raw_prompts, device)
        category_ids = torch.tensor(
            [[
                predictor.network.text_guided_adapter.CATEGORY_MAP.get(
                    case["categories"].get(str(i)),
                    0,
                )
                for i in sorted(map(int, case["findings"].keys()))
            ]],
            device=device,
            dtype=torch.long,
        )
        logits = predictor.predict_sliding_window_return_logits(
            preprocessed,
            embeddings,
            category_ids=category_ids,
            text_token_embeddings=text_token_embeddings,
            text_attention_mask=text_attention_mask,
        ).to("cpu")
        with torch.no_grad():
            prediction = (torch.sigmoid(logits.float()) > 0.5).numpy().astype(np.uint8)

        restored = np.zeros((prediction.shape[0], *original_shape), dtype=np.uint8)
        x_slice, y_slice, z_slice = bbox
        restored[:, x_slice[0]:x_slice[1], y_slice[0]:y_slice[1], z_slice[0]:z_slice[1]] = prediction

        output_subdir = case_output_dir / "raw_adapter"
        output_subdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for prompt_idx, prompt in enumerate(raw_prompts):
            output_path = output_subdir / f"{prompt_idx:02d}_{slugify(prompt)}.nii.gz"
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
