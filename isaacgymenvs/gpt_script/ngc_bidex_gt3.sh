#!/bin/bash

job_name="ngc_bidex_human_baseline"
tasks=("ShadowHandScissors" "ShadowHandSwingCup" "ShadowHandSwitch" "ShadowHandTwoCatchUnderarm")

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