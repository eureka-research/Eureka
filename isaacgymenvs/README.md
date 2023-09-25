# Jason Quickstart Doc
RewardGPT Curriculum:
```
python gpt_curriculum.py env=quadcopter iteration=1 sample=1 env.description="to make the quadrotor do a backflip"
```

Pen Spinning RLHF quickstart (using Eureka base policy):
```
python gpt_rlhf.py checkpoint="/data2/jasonyma/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/last_ShadowHandGPT_ep_20000.pth" max_iterations=500 sample=5 iteration=1 use_wandb=True env=shadow_hand env.description="spin the pen in sonic style. Please ignore goal attributes as they are not relevant for the pen style"
```

Pen Spinning RLHF quickstart (just generate reward candidates):
```
python gpt_rlhf.py max_iterations=5 sample=5 iteration=1 env=shadow_hand env.description="consistently spin and toss the pen in the air. Please ignore goal attributes as they are not relevant for the pen style"
```

Pen Spinning manual reward design:
```
python train_rl_gpt.py hydra.job.name="gpt_rlhf_test" task=ShadowHandGPT headless=False capture_video=True wandb_activate=True checkpoint="/data2/jasonyma/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/last_ShadowHandGPT_ep_20000.pth" max_iterations=100000 capture_video_freq=2000
```

<!-- Pen Spinning RLHF quickstart (using human base policy with a lot more training samples):
```
python gpt_rlhf.py checkpoint="/data2/jasonyma/isaac_gpt/train_rl_gpt/2023-09-04_15-27-10_shadow_pen_8gpu/runs/ShadowHand-2023-09-04_15-27-10/nn/last_ShadowHand_ep_20000.pth" max_iterations=500 sample=5 iteration=1 use_wandb=True env=shadow_hand env.description="spin the pen in sonic style. Please ignore goal attributes as they are not relevant for the pen style"
``` -->

Multi-GPU Training:
```
torchrun --nproc_per_node=2 train_rl_gpt.py task=ShadowHand checkpoint="/data2/jasonyma/isaac_gpt/train_rl_gpt/2023-09-04_15-27-10_shadow_pen_8gpu/runs/ShadowHand-2023-09-04_15-27-10/nn/last_ShadowHand_ep_20000.pth" multi_gpu=True max_iterations=1000 hydra.job.name="gpt_rlhf_test"
```

Multi-GPU Training (on NGC):
```
torchrun --nproc_per_node=8 train_rl_gpt.py task=ShadowHand multi_gpu=True 
```

Evaluating Fine-Tuned Pen Spinning (Eureka):
```
python train_rl_gpt.py test=True capture_video_len=1000 headless=False task=ShadowHandGPT checkpoint="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandGPT-2023-09-18_06-20-56.pth" hydra.job.name="gpt_rlhf_test" num_envs=64 force_render=True
```
```
python train_rl_gpt.py test=True capture_video_len=1000 headless=False task=ShadowHandUpsideDownGPT checkpoint="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandUpsideDownGPT-2023-09-18_13-56-42.pth" hydra.job.name="gpt_rlhf_test" num_envs=64 force_render=True
```
Evaluating Pre-trained Pen Spinning (Eureka):
```
python train_rl_gpt.py test=True capture_video=True capture_video_len=1000 headless=False task=ShadowHandSpin checkpoint="/data2/jasonyma/isaac_gpt/reward-gpt4-final-isaac-markov-gtfeedback-hard/last_ShadowHandGPT_ep_20000.pth" hydra.job.name="gpt_rlhf_test" num_envs=64
```
```
python train_rl_gpt.py test=True capture_video_len=1000 headless=False task=ShadowHand checkpoint="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandGPT-2023-09-12_20-19-09.pth" hydra.job.name="gpt_rlhf_test" num_envs=64 force_render=True
```
```
python train_rl_gpt.py test=True capture_video_len=1000 headless=False task=ShadowHandUpsideDownGPT checkpoint="/data2/jasonyma/isaac_gpt/gpt_policy/eureka-shadowhand-super-rl/ShadowHandUpsideDownGPT-2023-09-12_20-20-57.pth" hydra.job.name="gpt_rlhf_test" num_envs=64 force_render=True
```

Evaluating Pre-trained Pen Spinning:
```
python train_rl_gpt.py test=True capture_video_len=1000 headless=False task=ShadowHand checkpoint="/data2/jasonyma/isaac_gpt/train_rl_gpt/2023-09-05_07-23-27_shadow_oen_8_gpu_final/runs/ShadowHand-2023-09-05_07-23-28/nn/last_ShadowHand_ep_20000.pth" hydra.job.name="gpt_rlhf_test" num_envs=64 capture_video=True 
```
<!-- Pen Spinning RLHF quickstart:
```
python train_rl_gpt.py task=ShadowHand checkpoint="/data2/jasonyma/isaac_gpt/train_rl_gpt/2023-09-04_15-27-10_shadow_pen_8gpu/runs/ShadowHand-2023-09-04_15-27-10/nn/last_ShadowHand_ep_20000.pth" 
``` -->

=========================================================================
Train Policy with GPT Rewards:
```
python gpt_bidex.py env=humanoid
```
See ``/cfg_gpt`` to see a list of environments.

Text-only Chat debugging mode:
```
python gpt_bidex.py env=humanoid text_only=True iteration=10 sample=1 
```

In context RLHF:
```
python gpt_bidex.py env=humanoid human=True iteration=20 sample=1 temperature=1. capture_video=True max_iterations=1000
```

Automatic Policy Feedback:
```
python gpt_bidex.py env=shadow_hand_over iteration=1 sample=5 temperature=1. max_iterations=1000
python gpt_bidex.py env=shadow_hand_kettle iteration=1 sample=10 temperature=1. max_iterations=3000
```

Stand-alone script for training policies:
```
python train_rl_gpt.py task=Humanoid wandb_activate=True capture_video=True headless=False force_render=False
python train_rl_gpt.py capture_video=True headless=False force_render=False test=True task=Ant checkpoint="/data2/jasonyma/isaac_gpt/train_rl_gpt/2023-08-22_17-25-26/runs/Ant-2023-08-22_17-25-26/nn/last_Ant_ep_500.pth"
```

Stand-alone script for training policies (HumanoidAMP):
```
python train_rl_gpt.py task=HumanoidAMP train=HumanoidPPO wandb_activate=True capture_video=True headless=False force_render=False
```

## Adding New Environments
Add environment code to ``tasks``; add shorter version of environment code to ``gpt_utils/env_obs``. Add GPT prompting config to ``cfg_gpt/env``. Add environment config to ``cfg/task``; add PPO training config to ``cfg/train``.

## Converting Bidex environments to This Repo
Rename ``reset`` to ``reset_idx``
Replace ``asset_root`` in ``_create_envs``; comment out second appearance of ``asset_root``.
Change ``up_axis_idx`` definition. Change bunch of things in init.
Use the following as reference:
```
diff -u /home/exx/Projects/IsaacGymEnvs/isaacgymenvs/tasks/bidexhands/shadow_hand_over.py /home/exx/Projects/DexterousHands/bidexhands/tasks/shadow_hand_over.py
```

## Stream-line Process for Doing Programmatic Evaluation
See the following examples:
```python gpt_utils/bidex_prune_env.py```


## Old Stuff..
Stand-alone script for generating rollout gifs (Deprecated):
```
CUDA_VISIBLE_DEVICES=1 python rollout.py task=Humanoidcamera num_envs=1 headless=False checkpoint=POLICY_PATH
```

Stand-alone script for evaluating policies :
```
python train_rl_gpt.py test=True task=Humanoid checkpoint=POLICY_PATH
```

Evaluate Policy with Rollout Videos (Doesn't work well yet):
```
python train_rl_gpt.py task=Humanoid test=True capture_video=True headless=False capture_video_len=1000 num_envs=32 checkpoint=POLICY_PATH
```


## Installation
Install IsaacGym-Preview-4 using official instrution and then create ``rlgpu`` conda environment by running ``./create_conda_env_rlgpu.sh`` in the unzipped ``isaacgym`` repository. Then, install ``isaacgymenvs``, ``rlgames`` in this repository; also, ``openai``. 

<!-- Train Policy:
```
python train.py task=ShadowHand
python train.py task=ShadowHand wandb_activate=True wandb_entity=jma2020 wandb_project=issac_pen
python train.py task=ShadowHandSpin wandb_activate=True wandb_entity=jma2020 wandb_project=issac_pen
```

Visualize Trained Policy:
```
python train.py task=ShadowHandJason num_envs=1 test=True checkpoint=/home/exx/Projects/IsaacGymEnvs/isaacgymenvs/runs/ShadowHand_30-16-00-03/nn/ShadowHand.pth
```

## Collect Rollout
```
CUDA_VISIBLE_DEVICES=1 python rollout.py task=ShadowHandJason num_envs=1 headless=False
CUDA_VISIBLE_DEVICES=1 python rollout.py task=ShadowHandSpin num_envs=1 headless=False
kill -9 %1
``` -->