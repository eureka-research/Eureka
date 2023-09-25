#!/bin/bash

# check if the flag is passed
if [ $# -eq 0 ]; then
    echo "No flag provided. Use -a, -b, -c, or -d"
    exit 1
fi

# the first command line argument is temperature, the second is sample, the third is iteration
TEMPERATURE=1.0
SAMPLE=16
ITERATION=5
MODEL=gpt-4-0314

# function to run python script for each environment
run_script() {
    for env in "$@"; do
        python gpt_bidex.py env=$env temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt4-final'
    done
}

# check the flag value and execute corresponding command
case $1 in
    -a)
        run_script shadow_hand_kettle shadow_hand_door_close_outward shadow_hand_door_close_inward shadow_hand_door_open_outward shadow_hand_scissors
        ;;
    -b)
        run_script shadow_hand_door_open_inward shadow_hand_push_block shadow_hand_bottle_cap shadow_hand_pen shadow_hand_over
        ;;
    -c)
        run_script shadow_hand_grasp_and_place shadow_hand_switch shadow_hand_swing_cup shadow_hand_block_stack shadow_hand_lift_underarm
        ;;
    -d)
        run_script shadow_hand_catch_underarm shadow_hand_two_catch_underarm shadow_hand_catch_abreast shadow_hand_catch_over2underarm shadow_hand_re_orientation
        ;;
    *)
        echo "Invalid flag. Use -a, -b, -c, -d"
        exit 1
        ;;
esac
