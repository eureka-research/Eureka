# CUDA_VISIBLE_DEVICES=0 python gpt_bidex.py env=shadow_hand_block_stack hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=1 python gpt_bidex.py env=shadow_hand_bottle_cap hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=2 python gpt_bidex.py env=shadow_hand_catch_abreast hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=3 python gpt_bidex.py env=shadow_hand_catch_underarm hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=4 python gpt_bidex.py env=shadow_hand_door_close_inward hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=5 python gpt_bidex.py env=shadow_hand_door_close_outward hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=6 python gpt_bidex.py env=shadow_hand_door_open_inward hydra/output=ngc capture_video=False human=False & sleep 2;
# CUDA_VISIBLE_DEVICES=7 python gpt_bidex.py env=shadow_hand_door_open_outward hydra/output=ngc capture_video=False human=False;

CUDA_VISIBLE_DEVICES=0 python gpt_bidex.py env=shadow_hand_grasp_and_place hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=1 python gpt_bidex.py env=shadow_hand_kettle hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=2 python gpt_bidex.py env=shadow_hand_lift_underarm hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=3 python gpt_bidex.py env=shadow_hand_pen hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=4 python gpt_bidex.py env=shadow_hand_push_block hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=5 python gpt_bidex.py env=shadow_hand_re_orientation hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=6 python gpt_bidex.py env=shadow_hand_scissors hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=7 python gpt_bidex.py env=shadow_hand_swing_cup hydra/output=ngc capture_video=False human=False;

CUDA_VISIBLE_DEVICES=0 python gpt_bidex.py env=shadow_hand_switch hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=1 python gpt_bidex.py env=shadow_hand_two_catch_underarm hydra/output=ngc capture_video=False human=False & sleep 2;
CUDA_VISIBLE_DEVICES=2 python gpt_bidex.py env=shadow_hand_over hydra/output=ngc capture_video=False human=False & sleep 2;