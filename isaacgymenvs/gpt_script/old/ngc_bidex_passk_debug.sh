TEMPERATURE=1.0
SAMPLE=16
ITERATION=5

python gpt_bidex.py env=shadow_hand_kettle temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION hydra/output=ngc capture_video=False human=False
python gpt_bidex.py env=shadow_hand_scissors temperature=$TEMPERATURE sample=$SAMPLE iteration=$ITERATION hydra/output=ngc capture_video=False human=False