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
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
):
    """Setup LoRA for GRPO checkpoint loading"""
    print(f"🔧 Setting up LoRA for GRPO checkpoint (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
    
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
    
    print("✅ LoRA setup complete for GRPO checkpoint loading")
    return lora_config


def load_grpo_checkpoint(checkpoint_path: str):
    """Load GRPO checkpoint (simpler structure than DeepSpeed)"""
    checkpoint_dir = Path(checkpoint_path)
    
    # GRPO saves directly as pytorch_model.bin
    checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Could not find pytorch_model.bin in {checkpoint_dir}")
    
    print(f"📥 Loading GRPO checkpoint: {checkpoint_file}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        print("✅ Successfully loaded GRPO checkpoint")
    except Exception as e:
        print(f"❌ Failed to load with weights_only=False: {e}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        print("✅ Successfully loaded checkpoint with weights_only=True")
    
    print(f"📤 Checkpoint has {len(checkpoint)} parameters")
    return checkpoint


def save_grpo_ckpt(args):
    """Convert GRPO checkpoint to HuggingFace format"""
    
    # Load checkpoint
    print("🔄 Loading GRPO checkpoint...")
    state_dict = load_grpo_checkpoint(args.checkpoint_path)
    
    print("🔧 Building base DNALLMModel...")
    print(f"   • dna_is_evo2: {args.dna_is_evo2}")
    print(f"   • dna_model_finetune: {args.dna_model_finetune}")
    print(f"   • dna_embedding_layer: {args.dna_embedding_layer}")
    
    # Create model with CPU to avoid OOM
    device = torch.device("cpu")
    print(f"📍 Using device: {device} (CPU-only conversion to avoid OOM)")
    
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
    
    # Keep on CPU
    model = model.to(device)
    
    # Check vocabulary size compatibility
    print("🔧 Checking vocabulary size compatibility...")
    
    checkpoint_vocab_size = None
    for k in state_dict.keys():
        if "embed_tokens" in k and "weight" in k:
            checkpoint_vocab_size = state_dict[k].shape[0]
            print(f"📊 Found embedding key: {k} with vocab size: {checkpoint_vocab_size}")
            break
    
    if checkpoint_vocab_size:
        current_vocab_size = len(model.text_tokenizer)
        print(f"📊 Checkpoint vocab size: {checkpoint_vocab_size}")
        print(f"📊 Current vocab size: {current_vocab_size}")
        
        if current_vocab_size != checkpoint_vocab_size:
            print(f"⚠️  Vocab size mismatch! Checkpoint has {checkpoint_vocab_size}, model has {current_vocab_size}")
            print("🔧 Will resize embeddings to match checkpoint after LoRA preparation")
    
    # Setup LoRA
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Resize embeddings if needed
    if checkpoint_vocab_size:
        print("🔧 Post-LoRA: Ensuring embedding sizes match checkpoint...")
        
        try:
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
    
    # Load state dict directly (GRPO checkpoint already has correct keys)
    print(f"📊 Loading {len(state_dict)} parameters from GRPO checkpoint...")
    
    # Sample keys
    model_keys = set(model.state_dict().keys())
    sample_model_keys = list(model_keys)[:5]
    print("🔍 Sample model keys:")
    for key in sample_model_keys:
        print(f"   • {key}")
    
    sample_checkpoint_keys = list(state_dict.keys())[:5]
    print("🔍 Sample checkpoint keys:")
    for key in sample_checkpoint_keys:
        print(f"   • {key}")
    
    # Load with strict=False to handle any mismatches
    result = model.load_state_dict(state_dict, strict=False)
    
    print(f"📥 load_state_dict completed → missing {len(result.missing_keys)} | unexpected {len(result.unexpected_keys)}")
    
    if result.missing_keys:
        print("⚠️  Missing keys:")
        for key in result.missing_keys[:10]:
            print(f"   • {key}")
        if len(result.missing_keys) > 10:
            print(f"   ... and {len(result.missing_keys) - 10} more")
    
    if result.unexpected_keys:
        print("⚠️  Unexpected keys:")
        for key in result.unexpected_keys[:10]:
            print(f"   • {key}")
        if len(result.unexpected_keys) > 10:
            print(f"   ... and {len(result.unexpected_keys) - 10} more")
    
    # Merge LoRA adapters before saving
    if hasattr(model.text_model, "peft_config"):
        print("🔗 Merging LoRA adapters...")
        model.text_model = model.text_model.merge_and_unload()
        print("✅ LoRA adapters merged into base model")
    else:
        print("⚠️  No LoRA adapters found to merge")
    
    # Prepare save directory
    if os.path.exists(args.save_dir):
        is_model_dir = any(
            f in os.listdir(args.save_dir)
            for f in ["pytorch_model.bin", "model.safetensors", "config.json", "tokenizer.json"]
            if os.path.isfile(os.path.join(args.save_dir, f))
        )
        
        unsafe_patterns = [".git", "src", "bioreason", "__pycache__", "train_", "test_"]
        is_unsafe = any(pattern in args.save_dir.lower() for pattern in unsafe_patterns)
        
        if is_model_dir and not is_unsafe:
            print(f"🗑️  Removing existing model directory: {args.save_dir}")
            shutil.rmtree(args.save_dir)
        elif not is_unsafe:
            print(f"⚠️  Directory exists but doesn't look like a model directory. Will create alongside existing files.")
        else:
            print(f"🚫 Refusing to remove directory that may contain important files: {args.save_dir}")
            print("Please specify a different save directory or manually remove the existing one.")
            return
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"💾 Saving complete merged model to {args.save_dir}...")
    
    # Save text model and tokenizer (now merged, no LoRA)
    model.text_model.save_pretrained(args.save_dir)
    model.text_tokenizer.save_pretrained(args.save_dir)
    
    # Save DNA projection layer
    dna_projection_path = os.path.join(args.save_dir, "dna_projection.pt")
    torch.save(model.dna_projection.state_dict(), dna_projection_path)
    print(f"✅ DNA projection saved to {dna_projection_path}")
    
    # Save DNA model
    dna_model_path = os.path.join(args.save_dir, "dna_model")
    os.makedirs(dna_model_path, exist_ok=True)
    if hasattr(model.dna_model, "save_pretrained"):
        model.dna_model.save_pretrained(dna_model_path)
    else:
        torch.save(model.dna_model.state_dict(), os.path.join(dna_model_path, "pytorch_model.bin"))
    print(f"✅ DNA model saved to {dna_model_path}")
    
    print("✅ Complete merged model saved successfully!")
    print(f"📁 Model saved to: {args.save_dir}")
    
    # Save missing and unexpected keys log
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
    dna_params = sum(p.numel() for p in model.dna_model.parameters())
    print(f"✅ Saved model with {total_params/1e6:.1f}M parameters (text {text_params/1e6:.1f}M • DNA {dna_params/1e6:.1f}M)")
    
    model.eval()
    print(f"✅ Model loaded from: {args.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GRPO checkpoint to HuggingFace format for DNALLMModel."
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
        default=16,
        help="LoRA rank (GRPO default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (GRPO default: 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (GRPO default: 0.0)",
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
        help="Path to GRPO checkpoint directory (e.g. .../checkpoint-700/)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the converted HuggingFace model",
    )
    args = parser.parse_args()
    
    save_grpo_ckpt(args)


if __name__ == "__main__":
    main()

