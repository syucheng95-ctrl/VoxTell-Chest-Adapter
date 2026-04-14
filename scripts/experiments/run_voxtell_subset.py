import json
import time
from pathlib import Path

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from rex_prompt_tools import parse_finding, format_structured_prompt
from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


ROOT = Path(__file__).resolve().parent
DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_subset"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
OUTPUT_DIR = ROOT / "outputs" / "voxtell_subset"

SELECTED_CASES = [
    "train_13249_b_2.nii.gz",
    "train_13631_a_1.nii.gz",
    "train_13577_a_2.nii.gz",
]


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


def load_cases() -> list[dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    val_by_name = {item["name"]: item for item in data["val"]}
    return [val_by_name[name] for name in SELECTED_CASES]


def save_predictions(
    io: NibabelIOWithReorient,
    properties: dict,
    restored: np.ndarray,
    prompts: list[str],
    case_output_dir: Path,
    mode_name: str,
) -> list[dict]:
    mode_output_dir = case_output_dir / mode_name
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, prompt in enumerate(prompts):
        prompt_slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
        prompt_slug = "_".join(filter(None, prompt_slug.split("_")))[:80]
        output_path = mode_output_dir / f"{idx:02d}_{prompt_slug}.nii.gz"
        io.write_seg(restored[idx], str(output_path), properties)
        voxels = int(restored[idx].sum())
        summary.append({"index": idx, "prompt": prompt, "output": str(output_path), "voxels": voxels})
        print(f"[{mode_name}] Saved {output_path.name} voxels={voxels}")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Dataset dir: {DATASET_DIR}")

    predictor = VoxTellPredictor(model_dir=str(MODEL_DIR), device=device)
    io = NibabelIOWithReorient()

    for case in load_cases():
        case_name = case["name"]
        image_path = DATASET_DIR / "images_flat" / case_name
        case_output_dir = OUTPUT_DIR / case_name.replace(".nii.gz", "")
        case_output_dir.mkdir(parents=True, exist_ok=True)

        raw_prompts = [case["findings"][str(i)] for i in sorted(map(int, case["findings"].keys()))]
        parsed_prompts = [parse_finding(prompt) for prompt in raw_prompts]
        structured_prompts = [format_structured_prompt(parsed, style="templated") for parsed in parsed_prompts]

        print(f"\n=== Running {case_name} ===")
        print(f"Prompts: {len(raw_prompts)}")
        for idx, (raw_prompt, structured_prompt) in enumerate(zip(raw_prompts, structured_prompts)):
            print(f"[{idx}] raw={raw_prompt}")
            print(f"    structured={structured_prompt}")

        t0 = time.perf_counter()
        image, properties = io.read_images([str(image_path)])
        image = np.asarray(image)

        preprocessed, bbox, original_shape = predictor.preprocess(image)
        preprocess_sec = time.perf_counter() - t0
        combined_prompts = raw_prompts + structured_prompts

        t1 = time.perf_counter()
        embeddings = embed_text_on_cpu(predictor, combined_prompts, device)
        embed_sec = time.perf_counter() - t1

        t2 = time.perf_counter()
        logits = predictor.predict_sliding_window_return_logits(preprocessed, embeddings).to("cpu")
        infer_sec = time.perf_counter() - t2

        with torch.no_grad():
            prediction = (torch.sigmoid(logits.float()) > 0.5).numpy().astype(np.uint8)

        restored = np.zeros((prediction.shape[0], *original_shape), dtype=np.uint8)
        x_slice, y_slice, z_slice = bbox
        restored[:, x_slice[0]:x_slice[1], y_slice[0]:y_slice[1], z_slice[0]:z_slice[1]] = prediction
        split_idx = len(raw_prompts)
        all_mode_summaries = {
            "raw": save_predictions(
                io=io,
                properties=properties,
                restored=restored[:split_idx],
                prompts=raw_prompts,
                case_output_dir=case_output_dir,
                mode_name="raw",
            ),
            "structured": save_predictions(
                io=io,
                properties=properties,
                restored=restored[split_idx:],
                prompts=structured_prompts,
                case_output_dir=case_output_dir,
                mode_name="structured",
            ),
        }

        comparison = []
        for idx, parsed in enumerate(parsed_prompts):
            raw_item = all_mode_summaries["raw"][idx]
            structured_item = all_mode_summaries["structured"][idx]
            comparison.append(
                {
                    "index": idx,
                    "raw_prompt": raw_prompts[idx],
                    "structured_prompt": structured_prompts[idx],
                    "parsed_fields": parsed.to_dict(),
                    "raw_voxels": raw_item["voxels"],
                    "structured_voxels": structured_item["voxels"],
                    "voxel_delta": structured_item["voxels"] - raw_item["voxels"],
                    "raw_output": raw_item["output"],
                    "structured_output": structured_item["output"],
                }
            )

        (case_output_dir / "summary_comparison.json").write_text(
            json.dumps(
                {
                    "case": case_name,
                    "comparison": comparison,
                    "timing": {
                        "preprocess_sec": preprocess_sec,
                        "embed_sec": embed_sec,
                        "infer_sec": infer_sec,
                        "total_sec": preprocess_sec + embed_sec + infer_sec,
                        "prompt_count_raw": len(raw_prompts),
                        "prompt_count_combined": len(combined_prompts),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"Timing preprocess={preprocess_sec:.1f}s embed={embed_sec:.1f}s "
            f"infer={infer_sec:.1f}s total={preprocess_sec + embed_sec + infer_sec:.1f}s"
        )


if __name__ == "__main__":
    main()
