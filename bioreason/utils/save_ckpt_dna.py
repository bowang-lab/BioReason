import os
import shutil
import torch
from bioreason.models.dna_llm import DNALLMModel
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
from pathlib import Path
import argparse
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Register Evo2Tokenizer with transformers
register_evo2_tokenizer()


def _get_target_modules(model: DNALLMModel):
    """Get target modules for LoRA"""
    target_modules = []
    seen_names = set()
    
    for name, module in model.text_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            target_name = names[-1]
            
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
    
    return list(target_modules)


def _setup_lora_for_checkpoint_loading(
    model: DNALLMModel,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
):
    """Setup LoRA for checkpoint loading without full training preparation"""
    print(f"🔧 Setting up LoRA for checkpoint loading (rank={lora_rank}, alpha={lora_alpha})")
    
    # Get target modules
    target_modules = _get_target_modules(model)
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Prepare text model for LoRA
    model.text_model = prepare_model_for_kbit_training(model.text_model)
    model.text_model = get_peft_model(model.text_model, lora_config)
    
    print("✅ LoRA setup complete for checkpoint loading")
    return lora_config


class DeepSpeedCheckpointAnalyzer:
    """Analyzes and works with DeepSpeed checkpoint structure"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None

    def load_deepspeed_checkpoint(self):
        """Load the DeepSpeed model states checkpoint"""
        checkpoint_dir = Path(self.checkpoint_path)

        # Try different possible checkpoint files
        possible_files = [
            "checkpoint/mp_rank_00_model_states.pt",  # DeepSpeed model states
            "output_dir/pytorch_model.bin",  # Alternative format
            "pytorch_model.bin",  # Direct format
        ]

        checkpoint_file = None
        for filename in possible_files:
            full_path = checkpoint_dir / filename
            if full_path.exists():
                checkpoint_file = full_path
                break

        if checkpoint_file is None:
            raise FileNotFoundError(f"Could not find model checkpoint in {checkpoint_dir}")

        print(f"📥 Loading DeepSpeed checkpoint: {checkpoint_file}")

        # Load with weights_only=False for DeepSpeed compatibility
        try:
            self.checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            print("✅ Successfully loaded checkpoint")
        except Exception as e:
            print(f"❌ Failed to load with weights_only=False: {e}")
            # Try with weights_only=True as fallback
            self.checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            print("✅ Successfully loaded checkpoint with weights_only=True")

    def extract_model_state_dict(self):
        """Extract the clean model state dict from DeepSpeed checkpoint"""
        if self.checkpoint is None:
            raise ValueError("Checkpoint not loaded. Call load_deepspeed_checkpoint() first.")

        # Get the module state dict
        if "module" in self.checkpoint:
            state_dict = self.checkpoint["module"]
        else:
            state_dict = self.checkpoint

        print(f"📤 Extracting model state dict with {len(state_dict)} parameters...")
        return state_dict


def save_ckpt(args):
    # Use the unified loading pipeline from analysis.py (same as reason.py)
    print("🔄 Loading model via DeepSpeedCheckpointAnalyzer ...")
    analyzer = DeepSpeedCheckpointAnalyzer(args.checkpoint_path)
    analyzer.load_deepspeed_checkpoint()
    # Extract raw state dict and load with identical logic to reason.py to avoid key mismatches
    state_dict = analyzer.extract_model_state_dict()

    print("🔧 Building base DNALLMModel …")
    print(f"   • dna_is_evo2: {args.dna_is_evo2}")
    print(f"   • dna_model_finetune: {args.dna_model_finetune}")
    print(f"   • dna_embedding_layer: {args.dna_embedding_layer}")

    # Create a custom model using current DNALLM architecture
    model = DNALLMModel(
        text_model_name=args.text_model_name,
        dna_model_name=args.dna_model_name,
        cache_dir=args.cache_dir,
        max_length_dna=args.max_length_dna,
        max_length_text=args.max_length_text,
        text_model_finetune=True,
        dna_model_finetune=args.dna_model_finetune,
        dna_is_evo2=args.dna_is_evo2,
        dna_embedding_layer=args.dna_embedding_layer,
    )

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"📍 Model moved to {device}")

    # ------------------------------------------------------------------
    # CRITICAL: Check vocabulary size compatibility with checkpoint FIRST
    # ------------------------------------------------------------------
    print("🔧 Checking vocabulary size compatibility...")

    # First, let's check what vocab size the checkpoint expects
    # Handle PEFT-style embedding keys with different suffixes
    checkpoint_vocab_size = None
    for k in state_dict.keys():
        if "embed_tokens" in k and (
            "weight" in k or "original_module.weight" in k or "modules_to_save.default.weight" in k
        ):
            checkpoint_vocab_size = state_dict[k].shape[0]
            print(f"📊 Found embedding key: {k} with vocab size: {checkpoint_vocab_size}")
            break

    if checkpoint_vocab_size:
        current_vocab_size = len(model.text_tokenizer)
        print(f"📊 Checkpoint vocab size: {checkpoint_vocab_size}")
        print(f"📊 Current vocab size: {current_vocab_size}")

        # If vocab sizes don't match, we have a problem with special tokens
        if current_vocab_size != checkpoint_vocab_size:
            print(f"⚠️  Vocab size mismatch! Checkpoint has {checkpoint_vocab_size}, model has {current_vocab_size}")
            print("🔧 Will resize embeddings to match checkpoint after LoRA preparation")

    # ------------------------------------------------------------------
    # Setup LoRA for checkpoint loading so that all LoRA modules exist **before**
    # we load the checkpoint.
    # ------------------------------------------------------------------
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ------------------------------------------------------------------
    # CRITICAL: After LoRA prep, resize embeddings to match checkpoint
    # ------------------------------------------------------------------
    if checkpoint_vocab_size:
        print("🔧 Post-LoRA: Ensuring embedding sizes match checkpoint...")

        try:
            # Get current embedding size after LoRA prep
            if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
                actual_model = model.text_model.base_model.model
                current_embed_size = actual_model.model.embed_tokens.weight.shape[0]
                current_lm_head_size = actual_model.lm_head.weight.shape[0]
            else:
                current_embed_size = model.text_model.model.embed_tokens.weight.shape[0]
                current_lm_head_size = model.text_model.lm_head.weight.shape[0]

            print(f"📊 Current embed_tokens size: {current_embed_size}")
            print(f"📊 Current lm_head size: {current_lm_head_size}")
            print(f"📊 Target vocab size: {checkpoint_vocab_size}")

            if current_embed_size != checkpoint_vocab_size or current_lm_head_size != checkpoint_vocab_size:
                print(f"🔧 Resizing embeddings to match checkpoint ({checkpoint_vocab_size})")

                # Resize using the correct model reference
                if hasattr(model.text_model, "base_model") and hasattr(model.text_model.base_model, "model"):
                    model.text_model.base_model.model.resize_token_embeddings(checkpoint_vocab_size)
                else:
                    model.text_model.resize_token_embeddings(checkpoint_vocab_size)

                print(f"✅ Successfully resized embeddings to {checkpoint_vocab_size}")
            else:
                print("✅ Embedding sizes already match checkpoint")

        except Exception as e:
            print(f"⚠️  Error checking/resizing embeddings: {e}")
            print("🔧 Will attempt to load checkpoint anyway...")
    else:
        print("⚠️  Could not detect checkpoint vocabulary size")

    # ---- key remapping identical to reason.py ----
    def new_key(k: str) -> str:
        if k.startswith("model."):  # deepspeed save
            return k[6:]
        if k.startswith("_forward_module."):
            return k[len("_forward_module.") :]
        return k

    magic = {new_key(k): v for k, v in state_dict.items()}

    model_keys = set(model.state_dict().keys())
    print(f"📊 Model has {len(model_keys)} keys")
    print(f"📊 Checkpoint has {len(magic)} keys")

    # Print some sample model keys to understand the structure
    sample_model_keys = [k for k in model_keys if "text_model" in k][:5]
    print("🔍 Sample model keys:")
    for key in sample_model_keys:
        print(f"   • {key}")

    sample_checkpoint_keys = [k for k in magic.keys() if "text_model" in k][:5]
    print("🔍 Sample checkpoint keys:")
    for key in sample_checkpoint_keys:
        print(f"   • {key}")

    remapped = {}
    for k, v in magic.items():
        new_k = k

        # ------------------------------------------------------------------
        # FIXED KEY MAPPING - Handle the actual checkpoint structure
        # ------------------------------------------------------------------

        # 1) Strip the leading "model." that appears in DeepSpeed checkpoints
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]

        # 2) Handle PEFT-style embedding keys FIRST before general text model mapping
        # PEFT saves embeddings with special suffixes that need to be handled
        if "embed_tokens" in new_k or "lm_head" in new_k:
            # Handle PEFT embedding patterns:
            # text_model.base_model.model.model.embed_tokens.original_module.weight -> text_model.base_model.model.model.embed_tokens.weight
            # text_model.base_model.model.model.embed_tokens.modules_to_save.default.weight -> text_model.base_model.model.model.embed_tokens.weight

            if "original_module.weight" in new_k:
                # Use the original module weights
                new_k = new_k.replace(".original_module.weight", ".weight")
                print(f"🔧 Mapping original_module embedding: {k} -> {new_k}")
            elif "modules_to_save.default.weight" in new_k:
                # Use the trained/modified weights (prefer these over original)
                new_k = new_k.replace(".modules_to_save.default.weight", ".weight")
                print(f"🔧 Mapping modules_to_save embedding: {k} -> {new_k}")

            # Now apply standard text model mapping (only if not already in PEFT format)
            if new_k.startswith("text_model.") and not new_k.startswith("text_model.base_model.model."):
                suffix = new_k[len("text_model.") :]
                new_k = f"text_model.base_model.model.{suffix}"

        # 3) Map text model components - FIXED FOR CORRECT PEFT STRUCTURE!
        # The checkpoint may already have correct PEFT structure: text_model.base_model.model.*
        # Only apply transformation if it doesn't already have the correct structure
        elif new_k.startswith("text_model.") and not new_k.startswith("text_model.base_model.model."):
            suffix = new_k[len("text_model.") :]  # Get everything after "text_model."
            new_k = f"text_model.base_model.model.{suffix}"

        # 4) Map DNA components - remove duplicate "model." prefix
        elif new_k.startswith("dna_model."):
            # Keep as is - this should map to model.dna_model.*
            pass
        elif new_k.startswith("dna_projection."):
            # Keep as is - this should map to model.dna_projection.*
            pass

        # 5) Handle the case where we have both "model.dna_projection.*" and "dna_projection.*"
        # Prefer the version without "model." prefix to avoid duplication
        if k.startswith("model.dna_projection.") and f"dna_projection.{k.split('.')[-1]}" in magic:
            print(f"⚠️  Skipping duplicate key: {k} (using dna_projection version)")
            continue

        # 6) Skip if we already have this key (prefer modules_to_save over original_module)
        if new_k in remapped and "modules_to_save" in k:
            print(f"🔧 Replacing with modules_to_save version: {new_k}")
        elif new_k in remapped and "original_module" in k:
            print(f"⚠️  Skipping original_module (already have modules_to_save): {k}")
            continue

        # Move tensor to the same device as the model
        if isinstance(v, torch.Tensor):
            v = v.to(device)

        remapped[new_k] = v

    magic = remapped
    print(f"🔄 After key mapping: {len(magic)} keys")

    # After remapping keys, filter out 4-bit base layer weights
    filtered_magic = {}
    for k, v in magic.items():
        # Skip 4-bit quantized base layer weights and their metadata
        if '.base_layer.weight' in k and v.shape[-1] == 1:
            print(f"⚠️  Skipping 4-bit quantized weight: {k}")
            continue
        if any(x in k for x in ['.absmax', '.quant_map', '.quant_state', '.nested_absmax', '.nested_quant_map']):
            print(f"⚠️  Skipping quantization metadata: {k}")
            continue
        filtered_magic[k] = v

    magic = filtered_magic
    print(f"🔄 After filtering quantized weights: {len(magic)} keys")

    # Load the state dict
    result = model.load_state_dict(magic, strict=False)

    print(
        f"📥 load_state_dict completed → missing {len(result.missing_keys)} | unexpected {len(result.unexpected_keys)}"
    )
    if result.missing_keys:
        print("⚠️  Sample missing keys:", result.missing_keys[:5])
    if result.unexpected_keys:
        print("⚠️  Sample unexpected keys:", result.unexpected_keys[:5])

    # Check if LoRA was loaded and merge it BEFORE saving
    if hasattr(model.text_model, "peft_config"):
        print("🔗 Merging LoRA adapters...")
        model.text_model = model.text_model.merge_and_unload()
        print("✅ LoRA adapters merged into base model")
    else:
        print("⚠️  No LoRA adapters found to merge")

    # Safety check: only remove if it's clearly a checkpoint/model save directory
    if os.path.exists(args.save_dir):
        # Check if directory looks like a model save directory (contains model files)
        is_model_dir = any(
            f in os.listdir(args.save_dir)
            for f in [
                "pytorch_model.bin",
                "model.safetensors",
                "config.json",
                "tokenizer.json",
            ]
            if os.path.isfile(os.path.join(args.save_dir, f))
        )

        # Extra safety: don't remove if directory looks like a source code directory
        unsafe_patterns = [
            ".git",
            "src",
            "bioreason",
            "__pycache__",
            "train_",
            "test_",
        ]
        is_unsafe = any(pattern in args.save_dir.lower() for pattern in unsafe_patterns)

        if is_model_dir and not is_unsafe:
            print(f"🗑️ Removing existing model directory: {args.save_dir}")
            shutil.rmtree(args.save_dir)
        elif not is_unsafe:
            print(
                f"⚠️ Directory exists but doesn't look like a model directory. Will create alongside existing files: {args.save_dir}"
            )
        else:
            print(f"🚫 Refusing to remove directory that may contain important files: {args.save_dir}")
            print("Please specify a different save directory or manually remove the existing one.")
            return

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"💾 Saving complete merged model to {args.save_dir}...")

    # Move model back to CPU for saving to avoid memory issues
    model = model.cpu()

    # Save text model and tokenizer (now merged, no LoRA)
    model.text_model.save_pretrained(args.save_dir)
    model.text_tokenizer.save_pretrained(args.save_dir)

    # Save DNA projection layer
    dna_projection_path = os.path.join(args.save_dir, "dna_projection.pt")
    torch.save(model.dna_projection.state_dict(), dna_projection_path)
    print(f"✅ DNA projection saved to {dna_projection_path}")
    
    print("✅ Complete merged model saved successfully!")
    print(f"📁 Model saved to: {args.save_dir}")
    print("ℹ️  Note: DNA model not saved (frozen weights, load from original checkpoint)")

    # save the missing and unexpected keys to a file
    with open(os.path.join(args.save_dir, "missing_and_unexpected_keys.txt"), "w") as f:
        f.write("Missing keys:\n")
        for key in result.missing_keys:
            f.write(f"{key}\n")
        f.write("\nUnexpected keys:\n")
        for key in result.unexpected_keys:
            f.write(f"{key}\n")
    print(f"💾 Saved missing and unexpected keys to {os.path.join(args.save_dir, 'missing_and_unexpected_keys.txt')}")

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    # Handle Evo2 wrapper for parameter counting
    if model.dna_is_evo2:
        dna_params = sum(p.numel() for p in model.dna_model.model.parameters())
    else:
        dna_params = sum(p.numel() for p in model.dna_model.parameters())
    print(
        f"✅ Loaded model with {total_params/1e6:.1f}M parameters "
        f"(text {text_params/1e6:.1f}M • DNA {dna_params/1e6:.1f}M)"
    )

    model.eval()

    print(f"✅ Model loaded from: {args.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed ZeRO checkpoint to HuggingFace format for DNALLMModel."
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        required=True,
        help="Text model name or path (e.g. Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--dna_model_name",
        type=str,
        required=True,
        help="DNA model name or path (e.g. InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--max_length_text",
        type=int,
        default=512,
        help="Maximum length of text sequences",
    )
    parser.add_argument(
        "--max_length_dna",
        type=int,
        default=2048,
        help="Maximum length of DNA sequences",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--dna_is_evo2",
        action="store_true",
        default=False,
        help="Whether the DNA model is Evo2",
    )
    parser.add_argument(
        "--dna_embedding_layer",
        type=str,
        default=None,
        help="Evo2 layer name to extract (required when dna_is_evo2=True)",
    )
    parser.add_argument(
        "--dna_model_finetune",
        action="store_true",
        default=False,
        help="Whether to finetune the DNA model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to DeepSpeed ZeRO checkpoint directory (e.g. .../last.ckpt/)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the converted HuggingFace model",
    )
    args = parser.parse_args()

    save_ckpt(args)


if __name__ == "__main__":
    main()
