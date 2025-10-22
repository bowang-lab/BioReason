#!/bin/bash -l
#SBATCH -J dna_grpo_test_colocate
#SBATCH -p goodarzilab_gpu_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # one trainer per GPU
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# ----- user/env -----
USER="adibvafa"
ENV_NAME="bio"
export PATH="/home/$USER/miniconda/envs/$ENV_NAME/bin:$PATH"
source "/home/$USER/miniconda/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

: "${SLURM_NTASKS:=2}"
: "${SLURM_CPUS_PER_TASK:=8}"

cd "$HOME/BioReason"

# ----- data & project -----
export CACHE_DIR="/large_storage/goodarzilab/bioreason/cache_dir"
export SFT_CHECKPOINT="/large_storage/goodarzilab/bioreason/checkpoints/nt-500m-qwen3-4b-finetune-kegg-Qwen3-4B-20250511-190543/nt-500m-qwen3-4b-finetune-kegg-Qwen3-4B-epoch=03-val_loss_epoch=0.3599.ckpt/output_dir"
export WANDB_PROJECT="GRPO-DNA-LLM-test"

# ----- runtime (single node) -----
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL tuning: single-node NVLink/PCIe
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Disable cuMem to avoid extra memory usage
export NCCL_CUMEM_ENABLE=0

# PyTorch allocator - helps with OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# single-node rendezvous
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))

echo "[driver] host=$(hostname) MASTER=${MASTER_ADDR}:${MASTER_PORT} tasks=${SLURM_NTASKS}"
nvidia-smi -L || true

# ---------- launch ----------
# One process per GPU; Accelerate/Trainer will pick up WORLD/RANK from Slurm env.
srun --ntasks="$SLURM_NTASKS" \
     --cpu-bind=cores \
     --gpu-bind=single:1 \
     --label \
     --output=rank-%j-%t.log \
     bash -s <<'SRUN_PAYLOAD'
set -euo pipefail
echo "[rank ${SLURM_PROCID}] host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"

# DO NOT remap CUDA_VISIBLE_DEVICES here; srun already isolated one GPU/task.
# DDP-style env
export WORLD_SIZE="${SLURM_NTASKS}"
export RANK="${SLURM_PROCID}"
export LOCAL_RANK=0                # one visible GPU -> local ordinal is always 0

# (Optional) quick sanity check:
python - <<'PY'
import torch, os
print("cuda_available:", torch.cuda.is_available(),
      "| num_gpus:", torch.cuda.device_count(),
      "| device:", torch.cuda.current_device() if torch.cuda.is_available() else None,
      "| bf16_ok:", torch.tensor([0], dtype=torch.bfloat16, device="cuda").dtype if torch.cuda.is_available() else None)
PY

python -u train_grpo.py \
  --text_model_name "Qwen/Qwen3-4B" \
  --dna_model_name "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
  --cache_dir "$CACHE_DIR" \
  --sft_checkpoint "$SFT_CHECKPOINT" \
  --peft_ckpt False \
  --deepspeed grpo_trainer_lora_model/ds_config_stage2.json \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --max_steps 100 \
  --max_completion_length 800 \
  --num_generations 8 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --beta 0.0 \
  --run_name dna-llm-grpo-test \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --temperature 1 \
  --top_p 0.95 \
  --top_k 20 \
  --output_dir /large_storage/goodarzilab/bioreason/checkpoints/dna-llm-grpo \
  --save_strategy steps --save_steps 50 --save_total_limit 2 \
  --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --log_completions True \
  --use_vllm True \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.2 \
  --vllm_max_model_len 3000 \
  --vllm_ckpt "$SFT_CHECKPOINT" \
  --bf16 True
SRUN_PAYLOAD

