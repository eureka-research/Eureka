# Brax (https://github.com/google/brax)  

## How to run:  

* **Setup**

```bash
poetry install -E brax
poetry run pip install --upgrade "jax[cuda]==0.3.13" -f https://storage.googleapis.com/jax-releases/jax_releases.html
poetry run pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

* **Ant** ```poetry run python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml```
* **Humanoid** ```poetry run python runner.py --train --file rl_games/configs/brax/ppo_humanoid.yaml```

## Visualization of the trained policy:  
* **brax_visualization.ipynb**

## Results:  
* **Ant** fps step: 1692066.6 fps total: 885603.1  
![Ant](pictures/brax/brax_ant.jpg)  
* **Humanoid** fps step: 1244450.3 fps total: 661064.5  
![Humanoid](pictures/brax/brax_humanoid.jpg)  
* **ur5e** fps step: 1116872.3 fps total: 627117.0  
![Humanoid](pictures/brax/brax_ur5e.jpg)  


![Alt Text](pictures/brax/humanoid.gif)
![Alt Text](pictures/brax/ur5e.gif)