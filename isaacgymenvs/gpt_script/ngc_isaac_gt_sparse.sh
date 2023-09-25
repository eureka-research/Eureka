
job_name="ngc_isaac_human_baseline_sparse"
tasks=("Cartpole" "BallBalance" "Ant" "Humanoid" "Quadcopter")
seeds=(0 1 2 3 4)
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


tasks=("FrankaCabinet" "AllegroHand" "ShadowHand" "Anymal")
seeds=(0 1 2 3 4)
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