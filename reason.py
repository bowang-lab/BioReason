import os
import re

import pathlib
from argparse import ArgumentParser
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoProcessor,
)

from datasets import load_dataset, DatasetDict

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
#from unsloth import FastLanguageModel, is_bfloat16_supported

from bioreason.models.dna_llm import DNALLMModel
from bioreason.dna_modules import NucleotideDNAModule
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.trainer import DNALLMGRPOTrainer, DNALLMGRPOConfig
from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
register_evo2_tokenizer()

# Custom TrainerCallback to override the saving mechanism
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SaveWithPyTorchCallback(TrainerCallback):
    """Custom callback to save models with PyTorch's native save mechanism instead of safetensors"""
    def on_save(self, args, state, control, **kwargs):
        # Get the checkpoint folder
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # Save with PyTorch instead of safetensors
        checkpoint_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        model = kwargs.get("model")
        
        # Get model unwrapped from accelerator etc.
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        # Save using PyTorch directly
        torch.save(unwrapped_model.state_dict(), checkpoint_path)
        
        # DNALLMModel doesn't have a direct config attribute, so we need to save
        # the configs of its sub-models
        if hasattr(unwrapped_model, "text_model"):
            if hasattr(unwrapped_model.text_model, "config"):
                unwrapped_model.text_model.config.save_pretrained(checkpoint_folder)
            # Handle PEFT models which might have base_model
            elif hasattr(unwrapped_model.text_model, "base_model") and hasattr(unwrapped_model.text_model.base_model, "config"):
                unwrapped_model.text_model.base_model.config.save_pretrained(checkpoint_folder)
        
        # Print info about what's being saved
        print(f"Saved model checkpoint to {checkpoint_folder}")
        lora_params = [k for k in unwrapped_model.state_dict().keys() if "lora" in k]
        print(f"Checkpoint contains {len(lora_params)} LoRA parameters")
        
        # Signal that we've saved
        control.should_save = False
        return control

def _get_target_modules(model: DNALLMModel):
    # Apply LoRA to all linear layers in the text model
    target_modules = []

    # Get all unique linear layer names
    seen_names = set()
    for name, module in model.text.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            target_name = names[-1]  # Use the last part of the name

            # Skip output head but include all other linear layers
            if target_name != "lm_head" and target_name not in seen_names:
                target_modules.append(target_name)
                seen_names.add(target_name)

    # Add attention-specific layers
    attention_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]
    for pattern in attention_patterns:
        if pattern not in seen_names:
            target_modules.append(pattern)

    # Return all unique layer names to apply LoRA to all layers
    return list(target_modules)


def extract_xml_answer(text: str) -> str:
    # answer = text.split("<answer>")[-1]
    # answer = answer.split("</answer>")[0]
    answer = text.split("</think>")[-1]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_kegg_questions() -> Dataset:
    data = load_dataset('wanglab/kegg', 'default') # type: ignore
    example_dna_sequences = ["ATCTACATGCAT", "CAGCAGCTACAG", "CATCACATCGACATCGAC"]
    num_dna_sequences = 2 # TODO: Change to 2!

    data = data.map(lambda x: { # type: ignore
        'prompt': [
     
            {
                'role': 'user',
                'content': [
                    *({'type': 'dna', 'text': None} for _ in range(num_dna_sequences)),
                    {'type': 'text', 'text': x['question']},
                ],
            },
        ],
        'dna_sequences': [x['reference_sequence'], x['variant_sequence']],
        'answer': x['answer'],
    })  # type: ignore

    return data

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(question_prompt: str) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main') # type: ignore

    example_dna_sequences = ["ATCTACATGCAT", "CAGCAGCTACAG", "CATCACATCGACATCGAC"]
    data = data.map(lambda x: { # type: ignore
        'prompt': [ 

            {
                'role': 'user',
                'content': [
                    *({'type': 'dna', 'text': None} for _ in range(len(example_dna_sequences))),
                    {'type': 'text', 'text': 'Give me a short introduction to large language model.'}
                ]
            },
        ],
        'dna_sequences': [dna for dna in example_dna_sequences],
        'answer': extract_hash_answer(x['answer']),
    }) # type: ignore
    
    return data # type: ignore

def get_gsm8k_questions_old(question_prompt: str) -> Dataset:
    data = load_dataset('openai/gsm8k', 'main') # type: ignore

    example_dna_sequences = ["ATCTACATGCAT", "CAGCAGCTACAG", "CATCACATCGACATCGAC"]
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {
                'role': 'user',
                'content': [
                    *({'type': 'dna', 'text': None} for _ in range(len(example_dna_sequences))),
                    {'type': 'text', 'text': question_prompt.format(Question=x['question'])}
                ]
            },
        ],
        'dna_sequences': [dna for dna in example_dna_sequences],
        'answer': extract_hash_answer(x['answer']),
    }) # type: ignore
    
    return data # type: ignore

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # extracted_responses = [r.lower().replace("answer:", "").strip() for r in extracted_responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if a.lower() in r.lower() else 0.0 for r, a in zip(extracted_responses, answer[0])]

def less_than_4_reward_func(completions, **kwargs) -> List[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if len(r.split(' ')) <= 4 else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n.*?\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*.*?"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
        ],
    }

@dataclass
class GRPOModelConfig(ModelConfig):
    
    # "HuggingFaceTB/SmolLM-135M-Instruct"
    # "Qwen/Qwen2.5-0.5B-Instruct"
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "Model checkpoint for weights initialization."})
    dna_model_name_or_path: str = field(default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", metadata={"help": "Model checkpoint for weights initialization."})
    cache_dir: str = field(default=None, metadata={"help": "Path to model cache directory."})
    max_length_text: int = field(default=800, metadata={"help": "Maximum length of text sequences."})
    max_length_dna: int = field(default=800, metadata={"help": "Maximum length of DNA sequences, in groups of 6 nucleotides."})
    sft_checkpoint: str = field(default=None, metadata={"help": "Path to the checkpoint for SFT."})
    lora_r: int = field(default=32, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_modules_to_save: Optional[List[str]] = field(
        default="embed_tokens",
        metadata={"help": "Model layers to unfreeze & train."},
    )
    freeze_dna_modules: bool = False

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    dataset_name: str = field(default="wanglab/kegg", metadata={"help": "Dataset name with default."})
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: List[str] = field(
        #default_factory=lambda: ["accuracy", "format"],
        default_factory=lambda: ["xmlcount", "soft_format", "strict_format", "less_than_4", "correctness"],
        #metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'xmlcount', 'soft_format', 'strict_format', 'less_than_4', 'correctness'"},
    )
    # max_pixels: Optional[int] = field(
    #     default=12845056,
    #     metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    # )
    # min_pixels: Optional[int] = field(
    #     default=3136,
    #     metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    # )
    # task_type: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Choose task type: 'default', 'gui', ..."},
    # )



reward_funcs_registry = {
    # "accuracy": accuracy_reward,
    # "format": format_reward,
    "xmlcount": xmlcount_reward_func,
    "soft_format": soft_format_reward_func,
    "strict_format": strict_format_reward_func,
    "less_than_4": less_than_4_reward_func,
    "correctness": correctness_reward_func,
}

def get_vlm_module(model_name_or_path):
    if any(mini_name in model_name_or_path.lower() for mini_name in ["qwen", "smol"]):
        return NucleotideDNAModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    
def _get_target_modules(model):
    # Apply LoRA to all linear layers in the text model
    target_modules = []

    # Get all unique linear layer names
    seen_names = set()
    for name, module in model.text_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            target_name = names[-1]  # Use the last part of the name

            # Skip output head but include all other linear layers
            if target_name != "lm_head" and target_name not in seen_names:
                target_modules.append(target_name)
                seen_names.add(target_name)

    # Add attention-specific layers
    attention_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "query",
        "key",
        "value",
    ]
    for pattern in attention_patterns:
        if pattern not in seen_names:
            target_modules.append(pattern)

    # Return all unique layer names to apply LoRA to all layers
    return list(target_modules)


def _prep_for_training(model, training_args, dna_model_finetune: bool = False) -> LoraConfig:
    """
    Load and configure the DNALLMModel.
    """

    # Freeze DNA encoder parameters
    if dna_model_finetune:
        pass
    else:
        for param in model.dna_model.parameters():
            param.requires_grad = False

    target_modules = _get_target_modules(model)

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare text model for training
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)

    # Make projection layer trainable
    for param in model.dna_projection.parameters():
        param.requires_grad = True

    return lora_config

def main(script_args, training_args, model_args):

    print(training_args.output_dir)
    #pl.seed_everything(args.seed)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Initialize model
    # Load tokenizer for target text
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = DNALLMModel(
        text_model_name=model_args.model_name_or_path,
        dna_model_name=model_args.dna_model_name_or_path,
        cache_dir=model_args.cache_dir,
        max_length_text=model_args.max_length_text,
        max_length_dna=model_args.max_length_dna,
        text_model_finetune=True,
        dna_model_finetune=not model_args.freeze_dna_modules
    )

    # load checkpoint
    if model_args.sft_checkpoint is not None:
        print(f"Loading SFT checkpoint from {model_args.sft_checkpoint}")
        
        # Determine if it's a directory (PEFT format) or file (PyTorch state dict)
        is_directory = os.path.isdir(model_args.sft_checkpoint)
        
        if is_directory:
            # It's a PEFT checkpoint directory - load properly with PEFT
            from peft import PeftModel
            
            # First initialize the text model with PEFT
            print("Loading as PEFT checkpoint directory")
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
            
        else:
            # It's a PyTorch state dict file
            print("Loading as PyTorch state dict file")
            checkpoint = torch.load(model_args.sft_checkpoint)
            
            # replace model.text_model with text_model for all in state dict
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
                _prep_for_training(model, model_args, dna_model_finetune=model_args.freeze_dna_modules)
                
                # Print some diagnostic info about the keys
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(magic.keys())
                print(f"Model has {len(model_keys)} keys")
                print(f"Checkpoint has {len(checkpoint_keys)} keys")
                
                # Try to map LoRA keys more intelligently
                new_magic = {}
                for k, v in magic.items():
                    # Try different prefix mappings based on common patterns
                    if "base_model.model" in k and k not in model_keys:
                        new_k = k.replace("text_model.base_model.model", "text_model")
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue
                    
                    # Try removing common prefixes
                    if k.startswith("text_model.") and k not in model_keys:
                        new_k = "text_model.base_model.model." + k[len("text_model."):]
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue
                    
                    # Keep original key if no mapping found
                    new_magic[k] = v
                
                # Include missing target modules in diagnostic info
                magic = new_magic
                print(f"After key mapping: {len(magic)} keys")
                
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
                magic = {k.replace("dna_model", "dna_model"): v for k, v in magic.items()}
                
                # Fix the shared memory tensors issue by making a copy of weights
                for key in list(magic.keys()):
                    if 'lm_head.weight' in key:
                        magic[key] = magic[key].clone()
                
                # Load weights before setting up LoRA
                result = model.load_state_dict(magic, strict=False)
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")
                
                # Now prepare for LoRA training
                _prep_for_training(model, model_args, dna_model_finetune=model_args.freeze_dna_modules)
    else:
        # No checkpoint, just prepare for training
        _prep_for_training(model, model_args, dna_model_finetune=model_args.freeze_dna_modules)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # reward_funcs = [
    #         xmlcount_reward_func,
    #         soft_format_reward_func,
    #         strict_format_reward_func,
    #         int_reward_func,
    #         correctness_reward_func,
    #     ]
    print("reward_funcs:", reward_funcs)

    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template()


    dataset = get_kegg_questions()
    
    #dataset = get_gsm8k_questions(question_prompt)

    print(dataset)

    #print('ITEM ONE OF THE DATASET', dataset['train'][0])

    # Custom callback to handle saving with PyTorch's native mechanism
    custom_save_callback = SaveWithPyTorchCallback()

    # Initialize the GRPO trainer with custom callback
    trainer = DNALLMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        dna_module=vlm_module_cls(),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        callbacks=[custom_save_callback],  # Add our custom callback
    )

    # Set the trainer to save in PyTorch format instead of safetensors
    training_args.save_safetensors = False

    # Train and push the model to the Hub
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    # Train and push the model to the Hub
    trainer.train()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Ensure we use PyTorch's save mechanism instead of safetensors
    training_args.save_safetensors = False

    main(script_args, training_args, model_args)
    
    # parser.add_argument("--wandb_project", type=str, default="dna-text-finetune")
    # parser.add_argument("--wandb_entity", type=str, default="adibvafa")

    # args = parser.parse_args()
