# Deepmind Control (https://github.com/deepmind/dm_control)  

* I could not find any ppo deepmind_control benchmark. It is a first version only. Will be updated later.

## How to run:
* **Humanoid (Stand, Walk or Run)** 
```
poetry install -E envpool
poetry run pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run python runner.py --train --file rl_games/configs/dm_control/humanoid_walk.yaml
```

## Results:

* No tuning. I just run it on a couple of envs.
* I used 4000 epochs which is ~32M steps for almost all envs except HumanoidRun. But a few millions of steps was enough for the most of the envs.
* Deepmind used a pretty strange reward and training rules. A simple reward transformation: log(reward + 1) achieves best scores faster.

| Env           | Rewards       |
| ------------- | ------------- |
| Ball In Cup Catch  | 938  |
| Cartpole Balance  | 988  |
| Cheetah Run | 685  |
| Fish Swim  | 600  |
| Hopper Stand  | 557  |
| Humanoid Stand  | 653  |
| Humanoid Walk  | 621  |
| Humanoid Run  | 200  |
| Pendulum Swingup  | 706  |
| Walker Stand  | 907  |
| Walker Walk  | 917  |
| Walker Run  | 702  |
