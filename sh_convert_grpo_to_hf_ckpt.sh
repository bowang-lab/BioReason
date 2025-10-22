#!/bin/bash

# Convert GRPO checkpoint to HuggingFace format for DNA-LLM
# Usage: ./sh_convert_grpo_to_hf_ckpt.sh

# =============================================================================
# Configuration - GRPO Checkpoint Conversion
# =============================================================================

# Input checkpoint path (GRPO checkpoint)
CHECKPOINT_PATH="/large_storage/goodarzilab/bioreason/checkpoints/dna-llm-grpo/checkpoint-700"

# Output directory for HuggingFace format
SAVE_DIR="/large_storage/goodarzilab/bioreason/checkpoints/dna-llm-grpo/checkpoint-700-hf"

# Model configuration (same as training)
TEXT_MODEL_NAME="Qwen/Qwen3-4B"
DNA_MODEL_NAME="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

# Path configuration
CACHE_DIR="/large_storage/goodarzilab/bioreason/cache_dir"

# Training hyperparameters (matching GRPO training from test_grpo_dna.sh)
MAX_LENGTH_TEXT=512
MAX_LENGTH_DNA=2048
LORA_RANK=16        # GRPO used lora_r=16
LORA_ALPHA=32       # GRPO used lora_alpha=32
LORA_DROPOUT=0.0    # GRPO used lora_dropout=0

# DNA-specific settings (same as SFT)
DNA_IS_EVO2=False
DNA_EMBEDDING_LAYER=""  # Only needed for Evo2
DNA_MODEL_FINETUNE=False

# =============================================================================
# Validation
# =============================================================================
# Allow command line arguments to override defaults
if [ "$#" -ge 1 ]; then
    CHECKPOINT_PATH="$1"
fi

if [ "$#" -ge 2 ]; then
    SAVE_DIR="$2"
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [save_dir]"
    echo "Example: $0 /path/to/checkpoint-700 /path/to/output/hf_model"
    echo "Using default paths from configuration above..."
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

echo "Converting GRPO checkpoint to HuggingFace format..."
echo "Input: $CHECKPOINT_PATH"
echo "Output: $SAVE_DIR"
echo "LoRA config: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"

# =============================================================================
# Run conversion
# =============================================================================

cd /home/adibvafa/BioReason

# Build command with conditional flags
CMD="/home/adibvafa/miniconda/envs/bio/bin/python bioreason/utils/save_grpo_ckpt.py \
    --checkpoint_path \"$CHECKPOINT_PATH\" \
    --save_dir \"$SAVE_DIR\" \
    --text_model_name \"$TEXT_MODEL_NAME\" \
    --dna_model_name \"$DNA_MODEL_NAME\" \
    --cache_dir \"$CACHE_DIR\" \
    --max_length_text $MAX_LENGTH_TEXT \
    --max_length_dna $MAX_LENGTH_DNA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT"

# Add optional flags only if True
if [ "$DNA_IS_EVO2" = "True" ]; then
    CMD="$CMD --dna_is_evo2"
    if [ -n "$DNA_EMBEDDING_LAYER" ]; then
        CMD="$CMD --dna_embedding_layer \"$DNA_EMBEDDING_LAYER\""
    fi
fi

if [ "$DNA_MODEL_FINETUNE" = "True" ]; then
    CMD="$CMD --dna_model_finetune"
fi

# Execute the command
eval $CMD

if [ $? -eq 0 ]; then
    echo "✅ Conversion completed successfully!"
    echo "HuggingFace model saved to: $SAVE_DIR"
    echo ""
    echo "Saved components:"
    echo "  - Text model (merged): $SAVE_DIR/"
    echo "  - Tokenizer: $SAVE_DIR/"
    echo "  - DNA projection: $SAVE_DIR/dna_projection.pt"
    echo "  - DNA model: $SAVE_DIR/dna_model/"
    echo "  - Key mapping log: $SAVE_DIR/missing_and_unexpected_keys.txt"
else
    echo "❌ Conversion failed!"
    exit 1
fi

