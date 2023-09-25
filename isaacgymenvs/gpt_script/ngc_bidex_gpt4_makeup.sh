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

# check the flag value and execute corresponding command
case $1 in
    -a)
        python gpt_bidex.py --multirun env=shadow_hand_grasp_and_place,shadow_hand_grasp_and_place,shadow_hand_grasp_and_place,shadow_hand_re_orientation,shadow_hand_catch_over2underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt4-final-markov'
        ;;
    -b)
        python gpt_bidex.py --multirun env=shadow_hand_swing_cup,shadow_hand_lift_underarm,shadow_hand_block_stack,shadow_hand_door_open_inward,shadow_hand_two_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt4-final-markov'
        ;;
    -c)
        python gpt_bidex.py --multirun env=shadow_hand_catch_underarm,shadow_hand_catch_abreast,shadow_hand_door_open_outward,shadow_hand_scissors,shadow_hand_door_close_inward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt4-final-markov'
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c"
        exit 1
        ;;
esac