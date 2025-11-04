#!/usr/bin/env python
"""
Test script for DNA-LLM model combining train_grpo.py initialization 
with eval_kegg_dna_vllm.py evaluation method.
"""

import os
import sys
import torch
import argparse
from typing import Dict, Any
from dataclasses import dataclass, field

# Add the bioreason package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trl.data_utils import maybe_apply_chat_template
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

from bioreason.models.dna_llm import DNALLMModel, get_target_modules
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.dataset.kegg import format_kegg_for_dna_llm

@dataclass
class TestModelConfig:
    """Configuration for testing DNA-LLM model."""
    text_model_name: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "Model checkpoint for weights initialization."})
    dna_model_name: str = field(default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", metadata={"help": "Model checkpoint for weights initialization."})
    cache_dir: str = field(default=None, metadata={"help": "Path to model cache directory."})
    max_length_text: int = field(default=800, metadata={"help": "Maximum length of text sequences."})
    max_length_dna: int = field(default=800, metadata={"help": "Maximum length of DNA sequences, in groups of 6 nucleotides."})
    sft_checkpoint: str = field(default=None, metadata={"help": "Path to the checkpoint for SFT."})
    lora_r: int = field(default=32, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_modules_to_save: str = field(default="embed_tokens", metadata={"help": "Model layers to unfreeze & train."})
    dna_model_finetune: bool = False
    dna_projection_finetune: bool = True
    peft_ckpt: bool = False

def load_first_kegg_example() -> Dict[str, Any]:
    """
    Load the first example from KEGG test dataset.
    
    Returns:
        First test example formatted for DNA-LLM evaluation
    """
    print("Loading first KEGG example from HuggingFace...")
    
    # Load the dataset
    dataset = load_dataset('wanglab/kegg', 'default')
    test_data = dataset['test']
    
    # Get the first example
    first_example = test_data[0]
    print(f"Loaded first example: {first_example['question'][:100]}...")
    
    # Format example for DNA-LLM
    formatted_example = format_kegg_for_dna_llm(first_example, is_sft=False)
    
    print(f"Formatted example for DNA-LLM evaluation")
    return formatted_example

def _prep_for_training(
    model: DNALLMModel,
    model_args: TestModelConfig,
    dna_model_finetune: bool = False,
    dna_projection_finetune: bool = True
) -> LoraConfig:
    """
    Load and configure the DNA-LLM model for training.
    Based on train_grpo.py approach.
    """    
    # DNA encoder
    if dna_model_finetune:
        model.dna_model.train()
        print("DNA model is training")
        for param in model.dna_model.parameters():
            param.requires_grad = True
    else:
        model.dna_model.eval()
        print("DNA model is eval")
        for param in model.dna_model.parameters():
            param.requires_grad = False
    
    # DNA projection
    if dna_projection_finetune:
        model.dna_projection.train()
        print("DNA projection is training")
        for param in model.dna_projection.parameters():
            param.requires_grad = True
    else:
        model.dna_projection.eval()
        print("DNA projection is eval")
        for param in model.dna_projection.parameters():
            param.requires_grad = False

    # Text model: setup LoRA and set to train mode
    if model_args.lora_r == 0:
        # Text model: full finetune
        model.text_model.train()
        print("Text model is training")
        for param in model.text_model.parameters():
            param.requires_grad = True
        return None
    
    else:
        # Text model: setup LoRA and set to train mode
        target_modules = get_target_modules(model)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.text_model = prepare_model_for_kbit_training(model.text_model)
        model.text_model = get_peft_model(model.text_model, lora_config)
        model.text_model.train()

        return lora_config


def initialize_model(model_args: TestModelConfig) -> DNALLMModel:
    """
    Initialize the DNA-LLM model using train_grpo.py approach.
    
    Args:
        model_args: Model configuration arguments
        
    Returns:
        Initialized DNA-LLM model
    """
    print("Initializing DNA-LLM model using train_grpo.py approach...")
    
    # Initialize model
    model = DNALLMModel(
        text_model_name=model_args.text_model_name,
        dna_model_name=model_args.dna_model_name,
        cache_dir=model_args.cache_dir,
        max_length_text=model_args.max_length_text,
        max_length_dna=model_args.max_length_dna,
        text_model_finetune=True,
        dna_model_finetune=model_args.dna_model_finetune,
        device="cuda",
    )
    
    # Move the model to GPU
    model = model.to("cuda")
    model.text_model.config.use_cache = False

    # Load checkpoint if provided
    if model_args.sft_checkpoint is not None:
        print(f"Loading SFT checkpoint from {model_args.sft_checkpoint}")
        
        # Determine if it's a directory (PEFT format) or file (PyTorch state dict)
        is_directory = os.path.isdir(model_args.sft_checkpoint) 
        print(f"model_args.peft_ckpt: {model_args.peft_ckpt}")
        
        if is_directory and model_args.peft_ckpt:
            # First initialize the text model with PEFT
            print("Loading as PEFT checkpoint directory")
            from peft import PeftModel
            model.text_model = PeftModel.from_pretrained(
                model.text_model,
                model_args.sft_checkpoint,
                is_trainable=True
            )
            
            # Verify loaded adapters
            print("Loaded LoRA adapters:", model.text_model.active_adapter)
            
            # Optional: Merge weights into base model
            print("Merging SFT LoRA weights into base model...")
            model.text_model = model.text_model.merge_and_unload()
            print("Successfully merged SFT knowledge into base model")

        elif is_directory and not model_args.peft_ckpt:
            # Use HF from_pretrained
            model.text_model = AutoModelForCausalLM.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            print(f"CALLING PREP_FOR_TRAINING with dna_model_finetune: {model_args.dna_model_finetune}, dna_projection_finetune: {model_args.dna_projection_finetune}")
            lora_config = _prep_for_training(
                model,
                model_args,
                dna_model_finetune=model_args.dna_model_finetune,
                dna_projection_finetune=model_args.dna_projection_finetune
            )
            print("Successfully loaded SFT checkpoint")

        else:
            # It's a PyTorch state dict file
            print("Loading as PyTorch state dict file")
            checkpoint = torch.load(model_args.sft_checkpoint)
            
            # Replace model.text_model with text_model for all in state dict
            def new_key(k):
                if k.startswith("=model."): return k[6:]
                elif k.startswith("_forward_module."): return k[len("_forward_module."):]
                else: return k
            
            if "state_dict" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["state_dict"].items()}
            elif "module" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["module"].items()}
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                # Direct state dict - the checkpoint itself is the state dict
                print("Detected direct state dict format")
                magic = {new_key(k): v for k, v in checkpoint.items()}
            else:
                raise ValueError(f"Unsupported checkpoint format: {model_args.sft_checkpoint}")
            
            # Handle prefix mapping for different model architectures
            lora_prefix = False
            for key in magic.keys():
                if "lora" in key:
                    lora_prefix = True
                    break
            
            if lora_prefix:
                print("Detected LoRA weights in state dict")
                # First prepare model for LoRA training
                _prep_for_training(model, model_args, 
                                 dna_model_finetune=model_args.dna_model_finetune, 
                                 dna_projection_finetune=model_args.dna_projection_finetune)
                
                # Then load weights, allowing missing/extra keys
                result = model.load_state_dict(magic, strict=False)
                
                if len(result.unexpected_keys) > 0:
                    print(f"Sample unexpected keys: {result.unexpected_keys[:5]}")
                if len(result.missing_keys) > 0:
                    print(f"Sample missing keys: {result.missing_keys[:5]}")
                    
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")
            else:
                print("Standard weights detected - remapping keys")
                # Map keys to model structure
                magic = {k.replace("text_model", "text_model.base_model.model"): v for k, v in magic.items()}
                
                # Fix the shared memory tensors issue by making a copy of weights
                for key in list(magic.keys()):
                    if 'lm_head.weight' in key:
                        magic[key] = magic[key].clone()
                
                # Load weights before setting up LoRA
                result = model.load_state_dict(magic, strict=False)
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")
                
                # Now prepare for LoRA training
                _prep_for_training(model, model_args, 
                                 dna_model_finetune=model_args.dna_model_finetune, 
                                 dna_projection_finetune=model_args.dna_projection_finetune)
    
    else:
        # No checkpoint, just prepare for training
        print(f"CALLING PREP_FOR_TRAINING with dna_model_finetune: {model_args.dna_model_finetune}, dna_projection_finetune: {model_args.dna_projection_finetune}")
        _prep_for_training(model, model_args, 
                         dna_model_finetune=model_args.dna_model_finetune, 
                         dna_projection_finetune=model_args.dna_projection_finetune)

    # Move the model to GPU
    model = model.to("cuda")
    
    print("✅ Model initialized successfully!")
    return model


def evaluate_single_example(
    model: DNALLMModel,
    processor: DLProcessor,
    example: Dict[str, Any],
    generation_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single example using eval_kegg_dna_vllm.py method.
    
    Args:
        model: The DNA-LLM model
        processor: The DL processor for tokenization
        example: The example to evaluate
        generation_kwargs: Generation parameters
        
    Returns:
        Dictionary containing the evaluation result
    """
    print("Evaluating single example...")
    device = 'cuda'
    
    # Prepare prompt text and inputs via DLProcessor to duplicate DNA pad tokens
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
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if k not in ['stop']}
    outputs = model.generate(
        input_ids=prepared["input_ids"].to(device),
        attention_mask=prepared["attention_mask"].to(device),
        dna_tokenized=prepared.get("dna_tokenized").to(device),
        batch_idx_map=prepared.get("batch_idx_map"),
        **generation_kwargs
    )

    # Extract the generated text
    generated_text = processor.batch_decode(outputs[0], skip_special_tokens=True)
    # processor.batch_decode(prepared["input_ids"], skip_special_tokens=False)
    
    # Extract answer from generated text (look for "Answer:" pattern)
    predicted_answer = ""
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
        predicted_answer = answer_part.lower()
    
    # Get ground truth answer
    ground_truth = example["answer"].strip().lower()

    # Clean both
    for char in ".,!?\"'":
        predicted_answer = predicted_answer.replace(char, "")
        ground_truth = ground_truth.replace(char, "")

    # Determine if prediction is correct
    is_correct = ground_truth in predicted_answer
    print(f'Predicted answer: {predicted_answer}')
    print(f'Ground truth: {ground_truth}')
    print(f'Is correct: {is_correct}')
    
    return {
        'prompts_text': prompts_text,
        "generated_text": generated_text,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "dna_sequences": example["dna_sequences"],
        "question": example["prompt"][0]["content"][-1]["text"]  # Extract question text
    }


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test DNA-LLM with single example")
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name of the text model"
    )
    parser.add_argument(
        "--dna_model_name",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        help="Name of the DNA model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models"
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint for SFT"
    )
    parser.add_argument(
        "--peft_ckpt",
        action="store_true",
        help="Whether the checkpoint is in PEFT format"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA R value"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--dna_model_finetune",
        action="store_true",
        help="Whether to finetune DNA model"
    )
    parser.add_argument(
        "--dna_projection_finetune",
        action="store_true",
        default=True,
        help="Whether to finetune DNA projection"
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("DNA-LLM Test Script")
    print("="*80)
    print(f"Text model: {args.text_model_name}")
    print(f"DNA model: {args.dna_model_name}")
    print(f"SFT checkpoint: {args.sft_checkpoint}")
    print(f"PEFT checkpoint: {args.peft_ckpt}")
    print("="*80)
    
    try:
        # Create model configuration
        model_args = TestModelConfig(
            text_model_name=args.text_model_name,
            dna_model_name=args.dna_model_name,
            cache_dir=args.cache_dir,
            sft_checkpoint=args.sft_checkpoint,
            peft_ckpt=args.peft_ckpt,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            dna_model_finetune=args.dna_model_finetune,
            dna_projection_finetune=args.dna_projection_finetune
        )
        
        # Load first example
        example = load_first_kegg_example()
        
        # Initialize model
        model = initialize_model(model_args)
        
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
        
        print(f"\nStarting evaluation of single example...")
        print("Generation parameters:")
        for key, value in generation_kwargs.items():
            print(f"  {key}: {value}")
        
        # Evaluate example
        result = evaluate_single_example(
            model=model,
            processor=processor,
            example=example,
            generation_kwargs=generation_kwargs
        )
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"Generated text: {result['generated_text']}")
        print(f"Predicted answer: {result['predicted_answer']}")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Correct: {result['is_correct']}")
        print("="*80)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
