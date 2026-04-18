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
from voxtell.model.chest_text_guided_adapter import ChestTextGuidedAdapter
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

from train_voxtell_adapter import ADAPTER_STATE_NAME, build_adapter_network


DATASET_JSON = ROOT / "datasets" / "ReXGroundingCT_mirror_meta" / "dataset.json"
DATASET_DIR = ROOT / "datasets" / "ReXGroundingCT_subset"
MODEL_DIR = ROOT / "models" / "voxtell_v1.1"
PROBE_CASES_PATH = ROOT / "问题定位" / "v4_probe_cases.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "probe_v4_ablation"


def slugify(prompt: str) -> str:
    prompt_slug = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    return "_".join(filter(None, prompt_slug.split("_")))[:80]


def load_probe_case_names(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ordered: list[str] = []
    seen: set[str] = set()
    for items in data.values():
        for item in items:
            case_name = item["case"]
            if case_name not in seen:
                seen.add(case_name)
                ordered.append(case_name)
    return ordered


def load_case_lookup() -> dict[str, dict]:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    return {case["name"]: case for case in data["val"]}


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


def build_category_ids(case: dict, num_prompts: int, device: torch.device) -> torch.Tensor:
    category_map = ChestTextGuidedAdapter.CATEGORY_MAP
    ids = []
    for idx in range(num_prompts):
        code = case.get("categories", {}).get(str(idx))
        ids.append(category_map.get(code, category_map["other"]) if code else category_map["none"])
    return torch.tensor([ids], device=device, dtype=torch.long)


def infer_suppression_groups(state_dict: dict) -> int | None:
    weight = state_dict.get("text_guided_adapter.to_suppression.2.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def collect_prompt_suppression_means(network: torch.nn.Module) -> list[float] | None:
    adapter = getattr(network, "text_guided_adapter", None)
    suppression_tensor = getattr(adapter, "last_suppression_tensor", None) if adapter is not None else None
    if suppression_tensor is None:
        return None
    if suppression_tensor.ndim != 3:
        return None
    values = suppression_tensor.detach().float().mean(dim=-1).squeeze(0).cpu().tolist()
    return [float(value) for value in values]


def run_single_case(
    predictor: VoxTellPredictor,
    case: dict,
    output_dir: Path,
    mode: str,
    force: bool,
) -> None:
    case_name = case["name"]
    case_output_dir = output_dir / case_name.replace(".nii.gz", "")
    summary_path = case_output_dir / "summary_adapter_raw.json"
    if summary_path.exists() and not force:
        print(f"Skipping {case_name} ({mode})")
        return

    image_path = DATASET_DIR / "images_flat" / case_name
    raw_prompts = [case["findings"][str(i)] for i in sorted(map(int, case["findings"].keys()))]
    parsed_prompts = [parse_finding(prompt) for prompt in raw_prompts]

    io = NibabelIOWithReorient()
    image, properties = io.read_images([str(image_path)])
    image = np.asarray(image)
    preprocessed, bbox, original_shape = predictor.preprocess(image)
    embeddings = embed_text_on_cpu(predictor, raw_prompts, predictor.device)
    category_ids = build_category_ids(case, len(raw_prompts), predictor.device)

    with torch.inference_mode():
        logits = predictor.predict_sliding_window_return_logits(
            preprocessed,
            embeddings,
            category_ids=category_ids,
        ).to("cpu")
        suppression_means = collect_prompt_suppression_means(predictor.network)
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
        rows.append(
            {
                "index": prompt_idx,
                "category": case["categories"].get(str(prompt_idx)),
                "raw_prompt": prompt,
                "parsed_fields": parsed_prompts[prompt_idx].to_dict(),
                "voxels": int(restored[prompt_idx].sum()),
                "suppression_mean": None if suppression_means is None else suppression_means[prompt_idx],
                "mode": mode,
                "output": str(output_path),
            }
        )
        print(f"    saved {output_path.name} voxels={rows[-1]['voxels']}")

    summary = {
        "case": case_name,
        "mode": mode,
        "items": rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe-set ablation for v4 suppression.")
    parser.add_argument("--adapter-run-dir", type=Path, required=True)
    parser.add_argument("--probe-cases", type=Path, default=PROBE_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mode", choices=["full", "no_suppression"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-insertion-point", choices=["pre_decoder", "post_decoder"], default="pre_decoder")
    parser.add_argument("--adapter-num-groups", type=int, default=8)
    parser.add_argument("--adapter-residual-scale", type=float, default=0.1)
    parser.add_argument("--adapter-gate-cap", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = args.output_root / f"v4_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Adapter run dir: {args.adapter_run_dir}")
    print(f"Output dir: {output_dir}")

    adapter_state_path = args.adapter_run_dir / ADAPTER_STATE_NAME
    if not adapter_state_path.exists():
        raise FileNotFoundError(f"Missing adapter state: {adapter_state_path}")
    state_dict = torch.load(adapter_state_path, map_location=device, weights_only=False)
    suppression_groups = infer_suppression_groups(state_dict)
    predictor = VoxTellPredictor(model_dir=str(MODEL_DIR), device=device)
    predictor.network = build_adapter_network(
        device=device,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_insertion_point=args.adapter_insertion_point,
        adapter_num_groups=args.adapter_num_groups,
        adapter_suppression_groups=suppression_groups,
        adapter_residual_scale=args.adapter_residual_scale,
        adapter_gate_cap=args.adapter_gate_cap,
    )
    predictor.network.load_state_dict(state_dict, strict=False)
    predictor.network.eval()

    if args.mode == "no_suppression":
        adapter = predictor.network.text_guided_adapter
        if adapter is None:
            raise RuntimeError("Network has no text_guided_adapter; cannot ablate suppression.")
        adapter.suppression_strength = 0.0

    case_lookup = load_case_lookup()
    case_names = load_probe_case_names(args.probe_cases)
    print(f"Loaded {len(case_names)} probe cases")
    for idx, case_name in enumerate(case_names, start=1):
        print(f"[{idx}/{len(case_names)}] Processing {case_name}")
        run_single_case(
            predictor=predictor,
            case=case_lookup[case_name],
            output_dir=output_dir,
            mode=args.mode,
            force=args.force,
        )


if __name__ == "__main__":
    main()
