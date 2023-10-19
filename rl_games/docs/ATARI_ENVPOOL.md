# Atari with Envpool (https://envpool.readthedocs.io/en/latest/)  

## How to run:  
* **Pong** 

```
poetry install -E envpool
poetry run pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run python runner.py --train --file rl_games/configs/atari/ppo_pong_envpool.yaml
```

## Results:  
* **Pong-v5** 2 minutes training time to achieve 20+ score.
![Pong](pictures/atari_envpool/pong_envpool.png)  
* **Breakout-v3** 15 minutes training time to achieve 400+ score.
![Breakout](pictures/atari_envpool/breakout_envpool.png)  


