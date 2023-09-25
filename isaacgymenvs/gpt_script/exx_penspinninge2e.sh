MAX_ITERATIONS=60000 
JOB_NAME="exx-penspinninge2e"
# CHECKPOINT="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandGPT-2023-09-18_06-20-56.pth"
CHECKPOINT="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandGPT-2023-09-12_20-19-09.pth"

CUDA_VISIBLE_DEVICES=1 python train_rl_gpt.py task=ShadowHandGPT wandb_activate=True max_iterations=$MAX_ITERATIONS hydra.job.name=$JOB_NAME checkpoint=$CHECKPOINT 