#!/bin/bash

# check if the flag is passed
if [ $# -eq 0 ]; then
    echo "No flag provided. Use -a or -b"
    exit 1
fi

# the first command line argument is temperature, the second is sample, the third is iteration
TEMPERATURE=1
SAMPLE=16
ITERATION=5

MODEL=gpt-4-0314
JOB_NAME='reward-gpt4-final-bidex-markov-v2'
# JOB_NAME='reward-gpt4-final-bidex-markov-gtfeedback-temp0.2'

# MODEL=gpt-3.5-turbo-16k-0613
# JOB_NAME='reward-gpt35-final-bidex-markov-gtfeedback'

# MODEL=gpt-4-0613
# JOB_NAME='reward-gpt613-final-bidex-markov-gtfeedback'

# MODEL=gpt-4
# JOB_NAME='reward-gpt-0827-final-bidex-markov-gtfeedback'

# check the flag value and execute corresponding command

case $1 in
    -a)
        python gpt_bidex.py --multirun env=shadow_hand_swing_cup,shadow_hand_catch_abreast temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -b)
        python gpt_bidex.py --multirun env=shadow_hand_lift_underarm,shadow_hand_block_stack temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -c)
        python gpt_bidex.py --multirun env=shadow_hand_door_open_inward,shadow_hand_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -d)
        python gpt_bidex.py --multirun env=shadow_hand_door_open_outward,shadow_hand_scissors temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    -e)
        python gpt_bidex.py --multirun env=shadow_hand_door_close_inward,shadow_hand_two_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac

# case $1 in
#     -a)
#         python gpt_bidex.py --multirun env=shadow_hand_kettle,shadow_hand_door_close_outward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -b)
#         python gpt_bidex.py --multirun env=shadow_hand_door_open_inward,shadow_hand_push_block temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -c)
#         python gpt_bidex.py --multirun env=shadow_hand_grasp_and_place,shadow_hand_block_stack temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -d)
#         python gpt_bidex.py --multirun env=shadow_hand_catch_underarm,shadow_hand_two_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -e)
#         python gpt_bidex.py --multirun env=shadow_hand_door_close_inward,shadow_hand_bottle_cap temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -f)
#         python gpt_bidex.py --multirun env=shadow_hand_door_open_outward,shadow_hand_scissors temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -g)
#         python gpt_bidex.py --multirun env=shadow_hand_pen,shadow_hand_over temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -h) 
#         python gpt_bidex.py --multirun env=shadow_hand_switch,shadow_hand_lift_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -i)
#         python gpt_bidex.py --multirun env=shadow_hand_catch_over2underarm,shadow_hand_re_orientation temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     -j)
#         python gpt_bidex.py --multirun env=shadow_hand_swing_cup,shadow_hand_catch_abreast temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name=$JOB_NAME
#         ;;
#     *)
#         echo "Invalid flag. Use -a, -b, -c, -d"
#         exit 1
#         ;;
# esac
