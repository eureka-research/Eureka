#!/bin/bash

# check if the flag is passed
if [ $# -eq 0 ]; then
    echo "No flag provided. Use -a or -b"
    exit 1
fi

# the first command line argument is temperature, the second is sample, the third is iteration
TEMPERATURE=1.0
SAMPLE=16
ITERATION=5
MODEL=gpt-4-0314
JOB_NAME='reward-gpt4-final-isaac-markov-gtfeedback-hard'
case $1 in
    -a)
        python gpt_rl.py --multirun env=shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME original_path=\"/home/user/workspace/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/2023-09-06_20-45-06/0_env=shadow_hand,iteration=5,model=gpt-4-0314,sample=16,temperature=1.0\"
        ;;
    -b)
        python gpt_rl.py --multirun env=shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME original_path=\"/home/user/workspace/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/2023-09-06_20-46-03/0_env=shadow_hand,iteration=5,model=gpt-4-0314,sample=16,temperature=1.0\"
        ;;
    -c)
        python gpt_rl.py --multirun env=shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME original_path=\"/home/user/workspace/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/2023-09-06_20-52-08/0_env=shadow_hand,iteration=5,model=gpt-4-0314,sample=16,temperature=1.0\"
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac