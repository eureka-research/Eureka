#!/bin/bash

job_name="ngc_bidex_human_baseline_sparse"
seeds=(0 1 2 3 4)
num_gpus=8

tasks=("ShadowHandCatchOver2Underarm" "ShadowHandGraspAndPlace" "ShadowHandKettle" "ShadowHandLiftUnderarm")
job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))    
    done
done

wait 

tasks=("ShadowHandOver" "ShadowHandPen" "ShadowHandPushBlock" "ShadowHandReOrientation")
job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))    
    done
done

wait 

tasks=("ShadowHandScissors" "ShadowHandSwingCup" "ShadowHandSwitch" "ShadowHandTwoCatchUnderarm")
job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))    
    done
done

wait 

tasks=("ShadowHandBlockStack" "ShadowHandBottleCap" "ShadowHandCatchAbreast" "ShadowHandCatchUnderarm")
job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))    
    done
done

wait 

tasks=("ShadowHandDoorCloseInward" "ShadowHandDoorCloseOutward" "ShadowHandDoorOpenInward" "ShadowHandDoorOpenOutward")
job_count=0
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    for seed in "${seeds[@]}"; do
        gpu=$((job_count % num_gpus))
        CUDA_VISIBLE_DEVICES=$gpu python train_rl_gpt.py task=$task hydra.job.name=$job_name hydra/output=ngc seed=$seed & sleep 3
        job_count=$((job_count + 1))    
    done
done
