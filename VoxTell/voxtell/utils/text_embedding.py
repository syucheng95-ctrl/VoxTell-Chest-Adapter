from typing import Iterable

import torch


CATEGORY_LABELS = {
    "1a": "bronchial wall thickening",
    "1b": "bronchiectasis",
    "1c": "emphysema",
    "1d": "fibrosis or scarring",
    "1e": "infiltration or interstitial opacity",
    "1f": "pleural effusion or pleural thickening",
    "2a": "atelectasis",
    "2b": "consolidation",
    "2c": "ground glass opacity",
    "2d": "pulmonary nodule",
    "2e": "pulmonary mass",
    "2f": "pleural abnormality",
    "2g": "airway abnormality",
    "2h": "other focal lesion",
}


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def wrap_with_instruction(text_prompts):
    instruct = 'Given an anatomical term query, retrieve the precise anatomical entity and location it represents'

    instruct_text_prompts = []
    for text in text_prompts:
        instruct_text_prompts.append(f'Instruct: {instruct}\nQuery: {text}')
    return instruct_text_prompts


def category_code_to_prompt(category_code: str | None) -> str:
    if not category_code:
        return "chest CT finding with unspecified category"
    label = CATEGORY_LABELS.get(category_code, "other chest CT finding")
    return f"chest CT finding category {category_code}: {label}"


def build_combined_prompt(prompt: str, category_code: str | None) -> str:
    category_prompt = category_code_to_prompt(category_code)
    return f"{category_prompt}. Detailed finding description: {prompt}"


def build_text_representations(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    pooled = last_token_pool(last_hidden_states, attention_mask).float()
    token_sequences: list[torch.Tensor] = []
    token_masks: list[torch.Tensor] = []
    for hidden, mask in zip(last_hidden_states, attention_mask):
        valid_len = int(mask.sum().item())
        valid_len = max(valid_len, 1)
        token_sequences.append(hidden[:valid_len].float().cpu())
        token_masks.append(mask[:valid_len].to(dtype=torch.bool).cpu())
    return pooled, token_sequences, token_masks


def collate_text_sequences(
    token_sequences: Iterable[torch.Tensor],
    token_masks: Iterable[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_list = list(token_sequences)
    mask_list = list(token_masks)
    if not sequence_list:
        raise ValueError("Expected at least one token sequence")
    max_len = max(seq.shape[0] for seq in sequence_list)
    hidden_dim = sequence_list[0].shape[-1]
    batch = torch.zeros((1, len(sequence_list), max_len, hidden_dim), device=device, dtype=dtype)
    mask_batch = torch.zeros((1, len(sequence_list), max_len), device=device, dtype=torch.bool)
    for idx, (seq, mask) in enumerate(zip(sequence_list, mask_list)):
        length = seq.shape[0]
        batch[0, idx, :length] = seq.to(device=device, dtype=dtype)
        mask_batch[0, idx, :length] = mask.to(device=device, dtype=torch.bool)
    return batch, mask_batch
