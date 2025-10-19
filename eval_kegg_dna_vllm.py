#!/usr/bin/env python
"""
Evaluation script for DNA-vLLM model on KEGG test set.
Evaluates the vLLM-backed DNA-LLM model performance on biological reasoning tasks.
"""

import os
import sys
import json
import torch
import argparse
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Add the bioreason package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bioreason.models.dna_vllm import DNALLMModel
from bioreason.models.dl.processing_dl import DLProcessor
from trl.data_utils import maybe_apply_chat_template
from datasets import load_dataset, concatenate_datasets

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
            }
        ],
        "dna_sequences": [
            example["reference_sequence"],
            example["variant_sequence"],
        ],
        "answer": example["answer"],
    }

def load_kegg_test_dataset() -> List[Dict[str, Any]]:
    """
    Load the KEGG test dataset from HuggingFace.
    
    Returns:
        List of test examples formatted for DNA-LLM evaluation
    """
    print("Loading KEGG test dataset from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset('wanglab/kegg', 'default')
    test_data = dataset['test']

    # add in the val data as well to the mix
    val_data = dataset['val']
    
    test_data = concatenate_datasets([test_data, val_data])
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Format examples for DNA-LLM
    formatted_examples = []
    for example in test_data:
        formatted_example = format_kegg_for_dna_llm(example)
        formatted_examples.append(formatted_example)
    
    print(f"Formatted {len(formatted_examples)} examples for DNA-LLM evaluation")
    return formatted_examples


def initialize_model(
    ckpt_dir: str,
    text_model_name: str = "Qwen/Qwen3-4B",
    dna_model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    cache_dir: str = "/large_storage/goodarzilab/bioreason/cache_dir",
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 8192,
) -> DNALLMModel:
    """
    Initialize the DNA-vLLM model for evaluation.
    
    Args:
        ckpt_dir: Path to the checkpoint directory
        text_model_name: Name of the text model
        dna_model_name: Name of the DNA model
        cache_dir: Cache directory for models
        gpu_memory_utilization: GPU memory utilization for vLLM
        max_model_len: Maximum model length
        
    Returns:
        Initialized DNA-vLLM model
    """
    print("Initializing DNA-vLLM model...")
    
    # Check if checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Initialize the model
    model = DNALLMModel(
        ckpt_dir=ckpt_dir,
        text_model_name=text_model_name,
        dna_model_name=dna_model_name,
        cache_dir=cache_dir,
        max_length_dna=2048,
        max_length_text=512,
        text_model_finetune=False,
        dna_model_finetune=False,
        dna_is_evo2=False,
        dna_embedding_layer=None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    
    print("✅ Model initialized successfully!")
    return model


def evaluate_single_example(
    model: DNALLMModel,
    processor: DLProcessor,
    example: Dict[str, Any],
    generation_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single example and return the result.
    
    Args:
        model: The DNA-vLLM model
        processor: The DL processor for tokenization
        example: The example to evaluate
        generation_kwargs: Generation parameters
        
    Returns:
        Dictionary containing the evaluation result
    """
    # Prepare prompt text and inputs via DLProcessor to duplicate DNA pad tokens
    #breakpoint()
    prompts_text = [maybe_apply_chat_template(example, processor)["prompt"]]
    prepared = processor(
        text=prompts_text,
        batch_dna_sequences=[example["dna_sequences"]],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=model.max_length_text,
        max_length_dna=model.max_length_dna,
    )

    # Generate response
    outputs = model.generate(
        input_ids=prepared["input_ids"],
        attention_mask=prepared["attention_mask"],
        dna_tokenized=prepared.get("dna_tokenized"),
        batch_idx_map=prepared.get("batch_idx_map"),
        **generation_kwargs
    )

    # outputs = model.generate(
    #     input_ids=prepared["input_ids"],
    #     attention_mask=prepared["attention_mask"],
    #     dna_tokenized=None,
    #     batch_idx_map=None,
    #     **generation_kwargs
    # )

    # outputs = model.generate(input_ids=prepared["input_ids"], attention_mask=prepared["attention_mask"], dna_tokenized=None, batch_idx_map=None, **generation_kwargs)
    
    # Extract the generated text
    generated_text = outputs[0] if outputs else ""
    
    # Extract answer from generated text (look for "Answer:" pattern)
    predicted_answer = ""
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
        predicted_answer = answer_part.lower()
    
    # Get ground truth answer
    ground_truth = example["answer"].strip().lower()

    # clean both
    for char in ".,!?\"'":
        predicted_answer = predicted_answer.replace(char, "")
        ground_truth = ground_truth.replace(char, "")

    # Determine if prediction is correct
    #is_correct = predicted_answer == ground_truth
    is_correct = ground_truth in predicted_answer
    print('predicted_answer:', predicted_answer, 'ground_truth:', ground_truth, 'is_correct:', is_correct)
    
    #breakpoint()
    return {
        'prompts_text': prompts_text,
        "generated_text": generated_text,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "dna_sequences": example["dna_sequences"],
        "question": example["prompt"][0]["content"][-1]["text"]  # Extract question text
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics from the results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing calculated metrics
    """
    total_examples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    
    # Calculate accuracy
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
    
    # For binary classification metrics, we need to determine positive/negative labels
    # Get all unique ground truth answers
    all_answers = [r["ground_truth"] for r in results]
    unique_answers = list(set(all_answers))
    
    if len(unique_answers) == 2:
        # Binary classification case
        pos_label = unique_answers[0]  # Assume first label is positive
        neg_label = unique_answers[1]
        
        true_positives = sum(1 for r in results 
                           if r["ground_truth"] == pos_label and r["predicted_answer"] == pos_label)
        false_positives = sum(1 for r in results 
                            if r["ground_truth"] == neg_label and r["predicted_answer"] == pos_label)
        false_negatives = sum(1 for r in results 
                            if r["ground_truth"] == pos_label and r["predicted_answer"] == neg_label)
        true_negatives = sum(1 for r in results 
                           if r["ground_truth"] == neg_label and r["predicted_answer"] == neg_label)
        
        # Calculate precision, recall, and F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_examples": total_examples,
            "correct_predictions": correct_predictions,
            "positive_label": pos_label,
            "negative_label": neg_label
        }
    else:
        # Multi-class case - only accuracy is meaningful
        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_predictions": correct_predictions,
            "unique_labels": unique_answers
        }


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_dir: str,
    model_name: str = "dna_vllm"
) -> None:
    """
    Save evaluation results and metrics to files.
    
    Args:
        results: List of evaluation results
        metrics: Calculated metrics
        output_dir: Directory to save results
        model_name: Name of the model for file naming
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{model_name}_kegg_eval_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f"{model_name}_kegg_eval_metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    # Save results as CSV for easy analysis
    csv_file = os.path.join(output_dir, f"{model_name}_kegg_eval_results_{timestamp}.csv")
    df_data = []
    for i, result in enumerate(results):
        df_data.append({
            "example_id": i,
            "question": result["question"],
            "predicted_answer": result["predicted_answer"],
            "ground_truth": result["ground_truth"],
            "is_correct": result["is_correct"],
            "generated_text": result["generated_text"]
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    print(f"Results CSV saved to: {csv_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Positive label: {metrics['positive_label']}")
        print(f"Negative label: {metrics['negative_label']}")
    
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DNA-vLLM on KEGG test set")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Name of the text model"
    )
    parser.add_argument(
        "--dna_model_name",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        help="Name of the DNA model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/large_storage/goodarzilab/bioreason/cache_dir",
        help="Cache directory for models"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (currently only supports 1)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None for all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=800,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.4,
        help="GPU memory utilization for vLLM"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("DNA-vLLM KEGG Evaluation Script")
    print("="*80)
    print(f"Checkpoint directory: {args.ckpt_dir}")
    print(f"Text model: {args.text_model_name}")
    print(f"DNA model: {args.dna_model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max examples: {args.max_examples if args.max_examples else 'All'}")
    print("="*80)
    
    try:
        # Load test dataset
        test_examples = load_kegg_test_dataset()
        
        # Limit examples if specified
        if args.max_examples:
            test_examples = test_examples[:args.max_examples]
            print(f"Limited to {len(test_examples)} examples")
        
        # Initialize model
        model = initialize_model(
            ckpt_dir=args.ckpt_dir,
            text_model_name=args.text_model_name,
            dna_model_name=args.dna_model_name,
            cache_dir=args.cache_dir,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
        
        # Initialize processor
        processor = DLProcessor(
            tokenizer=model.text_tokenizer,
            dna_tokenizer=model.dna_tokenizer,
        )
        
        # Generation parameters
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "stop": ["<|im_end|>"],
        }
        
        print(f"\nStarting evaluation of {len(test_examples)} examples...")
        print("Generation parameters:")
        for key, value in generation_kwargs.items():
            print(f"  {key}: {value}")

        # Evaluate examples
        results = []
        for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
            result = evaluate_single_example(
                model=model,
                processor=processor,
                example=example,
                generation_kwargs=generation_kwargs
            )
            results.append(result)
            
            # Print progress every 10 examples
            if (i + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r["is_correct"])
                accuracy_so_far = correct_so_far / (i + 1)
                print(f"Progress: {i+1}/{len(test_examples)} | Accuracy so far: {accuracy_so_far:.4f}")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = calculate_metrics(results)
        
        # Save results
        save_results(results, metrics, args.output_dir)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
