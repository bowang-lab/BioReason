#!/bin/bash
#SBATCH --job-name=Qwen3_1.7B_SFT_RL    # Name of the job
#SBATCH --gres=gpu:4    # Number of GPUs
#SBATCH -p a100    # Partition
#SBATCH -c 12    # Number of cores
#SBATCH --time=12:00:00    # Time limit
#SBATCH --mem=128gb    # Memory limit
#SBATCH --output=Qwen3_1.7B_SFT_RL_a100-%j.out    # Output file
#SBATCH --error=Qwen3_1.7B_SFT_RL_a100-%j.err    # Error file

## Environment Setup
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "which python: $(which python)"

## Configuration Variables
# Change these to match your setup
SFT_CHECKPOINT=SFT_CHECKPOINT    # Change to the checkpoint of the SFT model
CACHE_DIR=CACHE_DIR    # Change to the directory where the model weights are cached
OUTPUT_DIR=OUTPUT_DIR    # Change to the directory where the model will be saved
CONDA_ENV=CONDA_ENV    # Change to the conda environment

## Setup Environment
conda activate $CONDA_ENV    # Change to the conda environment
cd .../BioReason/    # Change to the directory containing the script
nvidia-smi    # Check GPU status

## Dependencies
# You might need to install this on a gpu session
# pip install trl[vllm]

## =============================================================================
## Reinforcement Learning Training with DeepSpeed
## =============================================================================

# Run with DeepSpeed ZeRO Stage 2
srun deepspeed --num_gpus=4 --num_nodes=1 \
    reason.py \
    --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --bf16 true \
    --ddp_find_unused_parameters false \
    --sft_checkpoint $SFT_CHECKPOINT \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dna_model_name_or_path InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \
    --cache_dir $CACHE_DIR \
    --output_dir $OUTPUT_DIR \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --use_vllm true \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --num_train_epochs 1
