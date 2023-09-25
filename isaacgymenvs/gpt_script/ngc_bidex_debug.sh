# !/bin/bash

# check if the flag is passed
if [ $# -eq 0 ]; then
    echo "No flag provided. Use -a or -b"
    exit 1
fi

# the first command line argument is temperature, the second is sample, the third is iteration
JOB_NAME='reward-gpt4-markov-debug'
TEMPERATURE=1.0
SAMPLE=16
ITERATION=5
MODEL=gpt-4-0314
# REWARD_CRITERION='max'
case $1 in
    -a)
        python gpt_bidex.py --multirun env=shadow_hand_over,shadow_hand_over,shadow_hand_over,shadow_hand_over,shadow_hand_over reward_criterion=mean temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -b)
        python gpt_bidex.py --multirun env=shadow_hand_over,shadow_hand_over reward_criterion=max temperature=$TEMPERATURE sample=32 iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=shadow_hand_over,shadow_hand_over,shadow_hand_over reward_criterion=max temperature=$TEMPERATURE sample=32 iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -c)
        python gpt_bidex.py --multirun env=shadow_hand_over,shadow_hand_over,shadow_hand_over,shadow_hand_over,shadow_hand_over reward_criterion=max temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac