# Mujoco (https://github.com/deepmind/mujoco)  

## How to run:
* **Humanoid** 
```
poetry install -E envpool
poetry run pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run python runner.py --train --file rl_games/configs/mujoco/humanoid_envpool.yaml
```

## Results:
* **HalfCheetah-v4**
![HalfCheetah](pictures/mujoco/mujoco_halfcheetah_envpool.png)
* **Hopper-v4**  
![Hopper](pictures/mujoco/mujoco_hopper_envpool.png)
* **Walker2d-v4**  
![Walker2d](pictures/mujoco/mujoco_walker2d_envpool.png)
* **Ant-v4**
![Humanoid](pictures/mujoco/mujoco_ant_envpool.png)
* **Humanoid-v4**
![Humanoid](pictures/mujoco/mujoco_humanoid_envpool.png)
