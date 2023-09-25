#!/bin/bash

job_name="ngc_bidex_human_baseline"
tasks=("ShadowHandBlockStack" "ShadowHandBottleCap" "ShadowHandCatchAbreast" "ShadowHandCatchUnderarm" "ShadowHandDoorCloseInward" "ShadowHandDoorCloseOutward" "ShadowHandDoorOpenInward" "ShadowHandDoorOpenOutward")
seeds=(0)
num_gpus=8

job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))
    done
done

# Wait for all background jobs to finish
wait

seeds=(7 8 9)

job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))
    done
done
# CUDA_VISIBLE_DEVICES=0 python train_rl_gpt.py task=ShadowHandBlockStack hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=1 python train_rl_gpt.py task=ShadowHandBottleCap hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=2 python train_rl_gpt.py task=ShadowHandCatchAbreast hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=3 python train_rl_gpt.py task=ShadowHandCatchUnderarm hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=4 python train_rl_gpt.py task=ShadowHandDoorCloseInward hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=5 python train_rl_gpt.py task=ShadowHandDoorCloseOutward hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=6 python train_rl_gpt.py task=ShadowHandDoorOpenInward hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=7 python train_rl_gpt.py task=ShadowHandDoorOpenOutward hydra/output=ngc wandb_activate=True;