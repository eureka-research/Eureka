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
MODEL=gpt-3.5-turbo-16k-0613
JOB_NAME='reward-gpt4-final-isaac-markov-gtfeedback-gpt35'
case $1 in
    -a)
        python gpt_bidex.py --multirun env=cartpole,ballbalance,ant,humanoid,quadcopter temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=cartpole,ballbalance,ant,humanoid,quadcopter temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=cartpole,ballbalance,ant,humanoid,quadcopter temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=cartpole,ballbalance,ant,humanoid,quadcopter temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=cartpole,ballbalance,ant,humanoid,quadcopter temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -b)
        python gpt_bidex.py --multirun env=franka_cabinet,allegro_hand,anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=franka_cabinet,allegro_hand,anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=franka_cabinet,allegro_hand,anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -c)
        python gpt_bidex.py --multirun env=franka_cabinet,allegro_hand,anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        python gpt_bidex.py --multirun env=franka_cabinet,allegro_hand,anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        # python gpt_bidex.py --multirun env=anymal,shadow_hand temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac