# CUDA_VISIBLE_DEVICES=0 python train_rl_gpt.py task=Ant hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=1 python train_rl_gpt.py task=Anymal hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=2 python train_rl_gpt.py task=FrankaCabinet hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=3 python train_rl_gpt.py task=FrankaCubeStack hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=4 python train_rl_gpt.py task=Humanoid hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=5 python train_rl_gpt.py task=Ingenuity hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=6 python train_rl_gpt.py task=Quadcopter hydra/output=ngc wandb_activate=True & sleep 1;
# CUDA_VISIBLE_DEVICES=7 python train_rl_gpt.py task=ShadowHand hydra/output=ngc wandb_activate=True & sleep 1;

job_name="ngc_isaac_human_baseline"
# tasks=("Cartpole" "BallBalance" "Ant" "Humanoid" "Quadcopter")
# tasks=("FrankaCabinet" "FrankaCubeStack" "AllegroHand" "ShadowHand" "Anymal")
tasks=("BallBalance" "Trifinger")
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