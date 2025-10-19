import json
import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Tuple

from bioreason.dataset.utils import torch_to_hf_dataset
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dna_modules.nucleotide_module import NucleotideDNAModule
from trl.data_utils import maybe_apply_chat_template


class KEGGDataset(Dataset):
    """Dataset for KEGG data."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset by loading all JSON files from the given directory.

        Args:
            data_dir: Path to the directory containing JSON files
        """
        self.data_dir = data_dir
        self.data = []

        # Load all JSON files
        json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

        # Process each file
        for filename in json_files:
            file_path = os.path.join(data_dir, filename)
            kegg_id = filename.split("_")[1]

            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
                item["kegg_id"] = kegg_id
                processed_item = self._process_item(item)
                self.data.append(processed_item)

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data item to format fields as required.

        Args:
            item: Original data item from JSON

        Returns:
            Processed data item
        """
        # Extract question as is
        question = item.get("question", "")

        # Convert answer to lowercase and strip whitespace
        answer = item.get("answer", "").lower().strip()

        # Combine reasoning steps into a single paragraph with newlines
        reasoning_steps = item.get("reasoning", {}).get("reasoning_steps", [])
        reasoning = "\n".join(reasoning_steps)

        # Convert sequences to uppercase and strip whitespace
        reference_sequence = item.get("reference_sequence", "").upper().strip()
        variant_sequence = item.get("variant_sequence", "").upper().strip()

        return {
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "reference_sequence": reference_sequence,
            "variant_sequence": variant_sequence,
        }

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a specific item from the dataset."""
        return self.data[idx]


def split_kegg_dataset(
    dataset: KEGGDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[KEGGDataset, KEGGDataset, KEGGDataset]:
    """
    Split a KEGG dataset into train, validation, and test sets.

    Args:
        dataset: The dataset to split
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        batch_size: Batch size for the dataloaders
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Calculate the size of each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Set the random seed
    torch.manual_seed(seed)
    random.seed(seed)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset


def create_kegg_dataloader(
    data_dir: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the KEGG dataset.

    Args:
        data_dir: Path to the directory containing JSON files
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading data
        pin_memory: Whether to pin memory for faster data transfer

    Returns:
        DataLoader for the KEGG dataset
    """
    dataset = KEGGDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_format_kegg_function(model_name: str) -> Any:
    """
    Get the appropriate format function for a given model name.
    """
    if model_name.lower() == "llm":
        return format_kegg_for_llm
    elif model_name.lower() == "dna-llm":
        return format_kegg_for_dna_llm
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def format_kegg_for_dna_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for DNA-LLM.
    """
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": example["question"].strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            },
        ],
        "dna_sequences": [
            example["reference_sequence"],
            example["variant_sequence"],
        ],
        "answer": example["answer"],
    }


def format_kegg_for_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a KEGG example into the required chat format for LLM.
    """
    question = f"Reference sequence: {example['reference_sequence']}\nVariant sequence: {example['variant_sequence']}\nQuestion: {example['question']}"
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": question.strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            },
        ],
        "dna_sequences": [
            "",
            "",
        ],
        "answer": example["answer"],
    }

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
