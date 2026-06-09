#!/bin/bash

# Training script for Path Tracing - Latent 64x64
# Output image: 1024x1024 -> 64x64 latent grid (4096 tokens)
# Checkpoints saved to: ckpt/pt_latent64/<run_name>/

resume_from=${resume_from:-"models/BAGEL-7B-MoT"}
run_name=${run_name:-"run_8gpu"}
output_path=${output_path:-"./ckpt/pt_latent64/${run_name}/output"}
ckpt_path=${ckpt_path:-"./ckpt/pt_latent64/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set WANDB_API_KEY in your environment if you want to log to W&B
export WANDB_API_KEY="${WANDB_API_KEY:-}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/pt_ipt_8gpu.yaml \
  --model_path $resume_from \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune-from-ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_workers 8 \
  --max_latent_size 64 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 24576 \
  --expected_num_tokens 24576 \
  --mse_weight 1 \
  --ce_weight 1 \
  --save_every 1000 \
  --total_steps 25000
