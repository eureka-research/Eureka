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

# check the flag value and execute corresponding command
case $1 in
    -a)
        python gpt_bidex.py --multirun env=shadow_hand_kettle,shadow_hand_door_close_outward,shadow_hand_door_close_inward,shadow_hand_door_open_outward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        ;;
    -b)
        python gpt_bidex.py --multirun env=shadow_hand_scissors,shadow_hand_door_open_inward,shadow_hand_push_block,shadow_hand_bottle_cap temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        ;;
    *)
        echo "Invalid flag. Use -a or -b"
        exit 1
        ;;
esac

# python gpt_bidex.py --multirun env=shadow_hand_kettle,shadow_hand_scissors temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
# python gpt_bidex.py --multirun env=shadow_hand_door_close_outward,shadow_hand_door_close_inward,shadowhand_door_open_outward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
# python gpt_bidex.py --multirun env=shadow_hand_door_open_inward,shadow_hand_push_block,shadow_hand_bottle_cap temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'

# python gpt_bidex.py env=shadow_hand_kettle temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_scissors temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_door_close_outward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_door_close_inward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_door_open_outward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_door_open_inward temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_push_block temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_bottle_cap temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
