import torch
from typing import Any, Dict, List

from functools import partial

from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dna_modules.nucleotide_module import NucleotideDNAModule


def get_format_kegg_function(model_name: str, is_sft: bool=True) -> Any:
    """
    Get the appropriate format function for a given model name.
    """
    if model_name.lower() == "llm":
        return partial(format_kegg_for_llm, is_sft=is_sft)
    elif model_name.lower() == "dna-llm":
        return partial(format_kegg_for_dna_llm, is_sft=is_sft)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
def _format_kegg(example: Dict[str, Any], model_name: str, is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format.
    """

    if model_name.lower() not in ['llm', 'dna-llm']:
        raise ValueError(f"Unsupported model name: {model_name}")

    if model_name.lower() == 'llm':
        question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"
        reference_sequence = ""
        variant_sequence = ""
    elif model_name.lower() == 'dna-llm':
        question = example['question']
        reference_sequence = example['reference_sequence']
        variant_sequence = example['variant_sequence']
    
    item = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": question.strip()},
                ],
            }
        ],
        "dna_sequences": [
            reference_sequence,
            variant_sequence,
        ],
        "answer": example["answer"],
    }

    if is_sft:
        item['prompt'].append(
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            }
        )
    return item

def format_kegg_for_dna_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for DNA-LLM.
    """
    return _format_kegg(example, 'dna-llm', is_sft=is_sft)

def format_kegg_for_llm(example: Dict[str, Any], is_sft: bool) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for LLM.
    """
    return _format_kegg(example, 'llm', is_sft=is_sft)
    

def _truncate_after_assistant_start(text: str) -> str:
    """
    Keep everything up to and including the first '<|im_start|>assistant\n',
    drop any assistant answer that follows.
    """
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx != -1:
        return text[: idx + len(marker)]
    return text

def qwen_dna_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length_text: int,
    max_length_dna: int,
    return_answer_in_batch: bool = False,
    truncate_for_generation: bool = True
) -> Dict:
    """
    Custom collate function for Qwen DNA models.

    Creates a batch with proper labels for supervised fine-tuning where only
    the assistant responses contribute to the loss calculation.
    """

    dna_module = NucleotideDNAModule()
    # Keep original structured prompts for reward functions
    original_prompts = [example["prompt"] for example in examples]
    prompts_text = dna_module.prepare_prompt(processing_class=processor, inputs=examples)
    batch_dna_sequences = [example["dna_sequences"] for example in examples]

    batch = processor(
        text=prompts_text,
        batch_dna_sequences=batch_dna_sequences,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
    )

    # Create labels tensor filled with -100 (ignored in loss calculation)
    labels = torch.full_like(batch["input_ids"], -100)

    # Get token IDs for special markers
    assistant_start_marker = "<|im_start|>assistant\n"
    im_end_marker = "<|im_end|>"

    assistant_start_token_ids = processor.tokenizer.encode(
        assistant_start_marker, add_special_tokens=False
    )
    im_end_token_ids = processor.tokenizer.encode(
        im_end_marker, add_special_tokens=False
    )

    # Convert token arrays to tensors for faster comparison
    assistant_marker_tensor = torch.tensor(
        assistant_start_token_ids, device=batch["input_ids"].device
    )
    im_end_marker_tensor = torch.tensor(
        im_end_token_ids, device=batch["input_ids"].device
    )

    # Get dimensions for easier reference
    assistant_marker_len = len(assistant_start_token_ids)
    im_end_marker_len = len(im_end_token_ids)

    # For each sequence in the batch
    for i in range(batch["input_ids"].shape[0]):
        input_ids = batch["input_ids"][i]
        seq_len = input_ids.size(0)

        # Track assistant sections
        assistant_sections = []

        # Find all assistant start markers
        start_positions = []
        for pos in range(seq_len - assistant_marker_len + 1):
            if torch.all(
                input_ids[pos : pos + assistant_marker_len] == assistant_marker_tensor
            ):
                start_positions.append(
                    pos + assistant_marker_len
                )  # Store position after marker

        # Find all end markers
        end_positions = []
        for pos in range(seq_len - im_end_marker_len + 1):
            if torch.all(
                input_ids[pos : pos + im_end_marker_len] == im_end_marker_tensor
            ):
                end_positions.append(pos)  # Store position at start of end marker

        # Match start and end markers to create sections
        for start_pos in start_positions:
            # Find the next end marker after this start position
            valid_ends = [pos for pos in end_positions if pos > start_pos]
            if valid_ends:
                end_pos = min(valid_ends)  # Take the first end marker after start
                # Only include content between markers (not the markers themselves)
                if start_pos < end_pos:
                    assistant_sections.append((start_pos, end_pos))
            else:
                # If no end marker, assume the section runs to the end of the sequence
                assistant_sections.append((start_pos, seq_len))

        # Set labels for all identified assistant sections
        for start_pos, end_pos in assistant_sections:
            if start_pos < end_pos and start_pos < seq_len:
                end_pos = min(end_pos, seq_len)  # Safety check
                labels[i, start_pos:end_pos] = input_ids[start_pos:end_pos]

    # Also mask padding tokens
    labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    # Add labels to batch
    batch["labels"] = labels

    # Add answer to batch
    if return_answer_in_batch:
        batch["answer"] = [example["answer"].strip() for example in examples]

    prompts_text = [_truncate_after_assistant_start(p) for p in prompts_text]

    if truncate_for_generation:
        device = batch["input_ids"].device
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            # fall back to eos if pad is unset
            pad_id = processor.tokenizer.eos_token_id

        # composite marker to mirror _truncate_after_assistant_start
        composite = "<|im_end|>\n<|im_start|>assistant\n"
        comp_ids = processor.tokenizer.encode(composite, add_special_tokens=False)
        comp_t = torch.tensor(comp_ids, device=device)
        comp_len = len(comp_ids)

        B, L = batch["input_ids"].shape
        keep_lens: List[int] = []

        for i in range(B):
            ids = batch["input_ids"][i]
            keep = L  # default: keep all if marker not found
            # scan for FIRST occurrence to match your text truncation
            for j in range(0, L - comp_len + 1):
                if torch.all(ids[j:j+comp_len] == comp_t):
                    keep = j + comp_len
                    break
            keep_lens.append(keep)

        new_max = max(keep_lens) if keep_lens else 0

        # allocate new left-padded tensors
        new_input_ids = torch.full((B, new_max), pad_id, dtype=batch["input_ids"].dtype, device=device)
        new_attention = torch.zeros((B, new_max), dtype=batch["attention_mask"].dtype, device=device)
        new_labels = torch.full((B, new_max), -100, dtype=batch["labels"].dtype, device=device)

        for i, k in enumerate(keep_lens):
            if k == 0:
                continue
            # take the first k tokens (truncate from the RIGHT), then left-pad to new_max
            src_ids = batch["input_ids"][i, :k]
            src_attn = batch["attention_mask"][i, :k]
            src_lbls = batch["labels"][i, :k]

            new_input_ids[i, -k:] = src_ids
            new_attention[i, -k:] = src_attn
            new_labels[i, -k:] = src_lbls

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention
        batch["labels"] = new_labels

    batch["prompt"] = prompts_text
    batch["original_prompts"] = original_prompts

    return batch


def dna_collate_fn(
    batch: List[Dict[str, Any]],
    dna_tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Custom collate function for DNA models.
    """
    ref_sequences = [item["reference_sequence"] for item in batch]
    alt_sequences = [item["variant_sequence"] for item in batch]

    # Tokenize DNA sequences separately
    tokenized_ref = dna_tokenizer(
        ref_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized_alt = dna_tokenizer(
        alt_sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Get labels
    labels = []
    for item in batch:
        label = label2id[item["answer"]]
        labels.append(label)

    # Create labels tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    tokenized_batch = {
        "ref_ids": tokenized_ref.input_ids,
        "ref_attention_mask": tokenized_ref.attention_mask,
        "alt_ids": tokenized_alt.input_ids,
        "alt_attention_mask": tokenized_alt.attention_mask,
        "labels": labels_tensor,
    }

    return tokenized_batch
