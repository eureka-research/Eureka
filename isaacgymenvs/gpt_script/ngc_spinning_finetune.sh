JOB_NAME='eureka-shadowhand-super-rl'
MAX_ITERATIONS=600000
CHECKPOINT="/home/user/workspace/gpt_policy/eureka-shadowhand-super-rl/ShadowHandGPT-2023-09-12_20-19-09.pth"


# CUDA_VISIBLE_DEVICES=4 python train_rl_gpt.py task=ShadowHandGPT wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME checkpoint=$CHECKPOINT & sleep 3;
# CUDA_VISIBLE_DEVICES=5 python train_rl_gpt.py task=ShadowHandGPT wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME;

# CUDA_VISIBLE_DEVICES=6 python train_rl_gpt.py task=ShadowHandSpin wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME checkpoint=$CHECKPOINT & sleep 3;
# CUDA_VISIBLE_DEVICES=7 python train_rl_gpt.py task=ShadowHandSpin wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME;

case $1 in
    -a)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHandGPT multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME checkpoint=$CHECKPOINT
        ;;
    -b)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHandGPT multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac