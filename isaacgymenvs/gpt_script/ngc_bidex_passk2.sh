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
        python gpt_bidex.py --multirun env=shadow_hand_pen,shadow_hand_over,shadow_hand_grasp_and_place temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        python gpt_bidex.py --multirun env=shadow_hand_switch,shadow_hand_swing_cup,shadow_hand_block_stack temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        ;;
    -b)
        python gpt_bidex.py --multirun env=shadow_hand_lift_underarm,shadow_hand_catch_underarm,shadow_hand_two_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        python gpt_bidex.py --multirun env=shadow_hand_catch_abreast,shadow_hand_catch_over2underarm,shadow_hand_re_orientation temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc hydra.job.name='reward-gpt-final'
        ;;
    *)
        echo "Invalid flag. Use -a or -b"
        exit 1
        ;;
esac


# #!/bin/bash

# # the first command line argument is temperature, the second is sample, the third is iteration
# TEMPERATURE=1.0
# SAMPLE=32
# ITERATION=2
# MODEL=gpt-3.5-turbo-16k-0613

# python gpt_bidex.py env=shadow_hand_pen temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_over temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_grasp_and_place temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_switch temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_swing_cup temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_block_stack temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_lift_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_two_catch_underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_catch_abreast temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_two_over2underarm temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
# python gpt_bidex.py env=shadow_hand_re_orientation temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION model=$MODEL hydra/output=ngc capture_video=False human=False
