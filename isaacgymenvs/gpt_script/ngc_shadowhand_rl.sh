JOB_NAME='eureka-shadowhand-super-rl'
MAX_ITERATIONS=600000
case $1 in
    -a)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHandGPT multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -b)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHandUpsideDownGPT multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -c)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHand multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -d)
        torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHandUpsideDown multi_gpu=True wandb_activate=True max_iterations=$MAX_ITERATIONS hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac
