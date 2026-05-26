#!/bin/bash

# Training script for PET Mixed Answer-Only
# Resume from PET Latent64 25k checkpoint, train for 2k steps
# Dataloader mixing: 50% Visual CoT + 50% Answer-Only

resume_from=${resume_from:-"models/BAGEL-7B-MoT"}
run_name=${run_name:-"run_8gpu"}
output_path=${output_path:-"./ckpt/pet_mixed_answeronly/${run_name}/output"}
ckpt_path=${ckpt_path:-"./ckpt/pet_mixed_answeronly/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set WANDB_API_KEY in your environment if you want to log to W&B
export WANDB_API_KEY="${WANDB_API_KEY:-}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/pet_mixed_8gpu.yaml \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune-from-ema False \
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
  --save_every 500 \
  --total_steps 25000
