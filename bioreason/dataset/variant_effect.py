import json
import os
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Tuple


def get_format_variant_effect_function(model_name: str) -> Any:
    """
    Get the appropriate format function for a given model name.
    """
    if model_name.lower() == "llm":
        return format_variant_effect_for_llm
    elif model_name.lower() == "dna-llm":
        return format_variant_effect_for_dna_llm
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    

def clean_variant_effect_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a variant effect example.
    """
    example['answer'] = example['answer'].split(";")[0].strip().lower()
    return example


def clean_variant_effect_non_snv_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a variant effect non-SNV example.
    """
    example['answer'] = example['answer'].replace("[", "").replace("]", "").replace("'", "").replace("_", " ").strip()
    return example


def format_variant_effect_for_dna_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a VEP example into the required chat format for DNA-LLM.
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
                "reasoning_content": f"Answer: {example['answer'].strip()}",
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            },
        ],
        "dna_sequences": [
            example["reference_sequence"],
            example["variant_sequence"],
        ],
        "answer": example["answer"].strip(),
    }


def format_variant_effect_for_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a VEP example into the required chat format for LLM.
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
                "reasoning_content": f"Answer: {example['answer'].strip()}",
                "content": [
                    {"type": "text", "text": f"Answer: {example['answer'].strip()}"},
                ],
            },
        ],
        "dna_sequences": [
            "",
            "",
        ],
        "answer": example["answer"].strip(),
    }