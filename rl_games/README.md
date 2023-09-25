# RL Games: High performance RL library  

## Discord Channel Link 
* https://discord.gg/hnYRq7DsQh

## Papers and related links

* Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning: https://arxiv.org/abs/2108.10470
* DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality: https://dextreme.org/ https://arxiv.org/abs/2210.13702
* Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger: https://s2r2-ig.github.io/ https://arxiv.org/abs/2108.09779
* Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>
* Superfast Adversarial Motion Priors (AMP) implementation: https://twitter.com/xbpeng4/status/1506317490766303235 https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
* OSCAR: Data-Driven Operational Space Control for Adaptive and Robust Robot Manipulation: https://cremebrule.github.io/oscar-web/ https://arxiv.org/abs/2110.00704
* EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine: https://arxiv.org/abs/2206.10558 and https://github.com/sail-sg/envpool
* TimeChamber: A Massively Parallel Large Scale Self-Play Framework: https://github.com/inspirai/TimeChamber


## Some results on the different environments  

* [NVIDIA Isaac Gym](docs/ISAAC_GYM.md)

![Ant_running](https://user-images.githubusercontent.com/463063/125260924-a5969800-e2b5-11eb-931c-116cc90d4bbe.gif)
![Humanoid_running](https://user-images.githubusercontent.com/463063/125266095-4edf8d00-e2ba-11eb-9c1a-4dc1524adf71.gif)

![Allegro_Hand_400](https://user-images.githubusercontent.com/463063/125261559-38373700-e2b6-11eb-80eb-b250a0693f0b.gif)
![Shadow_Hand_OpenAI](https://user-images.githubusercontent.com/463063/125262637-328e2100-e2b7-11eb-99af-ea546a53f66a.gif)

* [Dextreme](https://dextreme.org/)

![Allegro_Hand_real_world](https://user-images.githubusercontent.com/463063/216529475-3adeddea-94c3-4ac0-99db-00e7df4ba54b.gif)

* [Starcraft 2 Multi Agents](docs/SMAC.md)  
* [BRAX](docs/BRAX.md)  
* [Mujoco Envpool](docs/MUJOCO_ENVPOOL.md) 
* [DeepMind Envpool](docs/DEEPMIND_ENVPOOL.md) 
* [Atari Envpool](docs/ATARI_ENVPOOL.md) 
* [Random Envs](docs/OTHER.md)  


Implemented in Pytorch:

* PPO with the support of asymmetric actor-critic variant
* Support of end-to-end GPU accelerated training pipeline with Isaac Gym and Brax
* Masked actions support
* Multi-agent training, decentralized and centralized critic variants
* Self-play 

 Implemented in Tensorflow 1.x (was removed in this version):

* Rainbow DQN
* A2C
* PPO

## Quickstart: Colab in the Cloud

Explore RL Games quick and easily in colab notebooks:

* [Mujoco training](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/mujoco_envpool_training.ipynb) Mujoco envpool training example.
* [Brax training](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/brax_training.ipynb) Brax training example, with keeping all the observations and actions on GPU.
* [Onnx discrete space export example with Cartpole](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_discrete.ipynb) envpool training example.
* [Onnx continuous space export example with Pendulum](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_continuous.ipynb) envpool training example.
* [Onnx continuous space with LSTM export example with Pendulum](https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_lstm_continuous.ipynb) envpool training example.

## Installation

For maximum training performance a preliminary installation of Pytorch 1.9+ with CUDA 11.1+ is highly recommended:

```conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -c nvidia``` or:
```pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html```

Then:

```pip install rl-games```

To run CPU-based environments either Ray or envpool are required ```pip install envpool``` or ```pip install ray```
To run Mujoco, Atari games or Box2d based environments training they need to be additionally installed with ```pip install gym[mujoco]```, ```pip install gym[atari]``` or ```pip install gym[box2d]``` respectively.

To run Atari also ```pip install opencv-python``` is required. In addition installation of envpool for maximum simulation and training perfromance of Mujoco and Atari environments is highly recommended: ```pip install envpool```

## Citing

If you use rl-games in your research please use the following citation:

```bibtex
@misc{rl-games2021,
title = {rl-games: A High-performance Framework for Reinforcement Learning},
author = {Makoviichuk, Denys and Makoviychuk, Viktor},
month = {May},
year = {2021},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/Denys88/rl_games}},
}
```


## Development setup

```bash
poetry install
# install cuda related dependencies
poetry run pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Training
**NVIDIA Isaac Gym**

Download and follow the installation instructions of Isaac Gym: https://developer.nvidia.com/isaac-gym  
And IsaacGymEnvs: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

*Ant*

```python train.py task=Ant headless=True```
```python train.py task=Ant test=True checkpoint=nn/Ant.pth num_envs=100```

*Humanoid*

```python train.py task=Humanoid headless=True```
```python train.py task=Humanoid test=True checkpoint=nn/Humanoid.pth num_envs=100```

*Shadow Hand block orientation task*

```python train.py task=ShadowHand headless=True```
```python train.py task=ShadowHand test=True checkpoint=nn/ShadowHand.pth num_envs=100```

**Other**

*Atari Pong*

```bash
poetry install -E atari
poetry run python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml
poetry run python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth
```

*Brax Ant*

```bash
poetry install -E brax
poetry run pip install --upgrade "jax[cuda]==0.3.13" -f https://storage.googleapis.com/jax-releases/jax_releases.html
poetry run python runner.py --train --file rl_games/configs/brax/ppo_ant.yaml
poetry run python runner.py --play --file rl_games/configs/brax/ppo_ant.yaml --checkpoint runs/Ant_brax/nn/Ant_brax.pth
```

## Experiment tracking

rl_games support experiment tracking with [Weights and Biases](https://wandb.ai).

```bash
poetry install -E atari
poetry run python runner.py --train --file rl_games/configs/atari/ppo_breakout_torch.yaml --track
WANDB_API_KEY=xxxx poetry run python runner.py --train --file rl_games/configs/atari/ppo_breakout_torch.yaml --track
poetry run python runner.py --train --file rl_games/configs/atari/ppo_breakout_torch.yaml --wandb-project-name rl-games-special-test --track
poetry run python runner.py --train --file rl_games/configs/atari/ppo_breakout_torch.yaml --wandb-project-name rl-games-special-test -wandb-entity openrlbenchmark --track
```


## Multi GPU

We use `torchrun` to orchestrate any multi-gpu runs.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 runner.py --train --file rl_games/configs/ppo_cartpole.yaml
```

## Config Parameters

| Field                  | Example Value             | Default | Description                                                                                                                                                  |
| ---------------------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| seed                   | 8                         | None    | Seed for pytorch, numpy etc.                                                                                                                                 |
| algo                   |                           |         | Algorithm block.                                                                                                                                             |
| name                   | a2c_continuous            | None    | Algorithm name. Possible values are: sac, a2c_discrete, a2c_continuous                                                                                       |
| model                  |                           |         | Model block.                                                                                                                                                 |
| name                   | continuous_a2c_logstd     | None    | Possible values: continuous_a2c ( expects sigma to be (0, +inf), continuous_a2c_logstd  ( expects sigma to be (-inf, +inf), a2c_discrete, a2c_multi_discrete |
| network                |                           |         | Network description.                                                                                                                                         |
| name                   | actor_critic              |         | Possible values: actor_critic or soft_actor_critic.                                                                                                          |
| separate               | False                     |         | Whether use or not separate network with same same architecture for critic. In almost all cases if you normalize value it is better to have it False         |
| space                  |                           |         | Network space                                                                                                                                                |
| continuous             |                           |         | continuous or discrete                                                                                                                                       |
| mu_activation          | None                      |         | Activation for mu. In almost all cases None works the best, but we may try tanh.                                                                             |
| sigma_activation       | None                      |         | Activation for sigma. Will be threated as log(sigma) or sigma depending on model.                                                                            |
| mu_init                |                           |         | Initializer for mu.                                                                                                                                          |
| name                   | default                   |         |                                                                                                                                                              |
| sigma_init             |                           |         | Initializer for sigma. if you are using logstd model good value is 0.                                                                                        |
| name                   | const_initializer         |         |                                                                                                                                                              |
| val                    | 0                         |         |                                                                                                                                                              |
| fixed_sigma            | True                      |         | If true then sigma vector doesn't depend on input.                                                                                                           |
| cnn                    |                           |         | Convolution block.                                                                                                                                           |
| type                   | conv2d                    |         | Type: right now two types supported: conv2d or conv1d                                                                                                        |
| activation             | elu                       |         | activation between conv layers.                                                                                                                              |
| initializer            |                           |         | Initialier. I took some names from the tensorflow.                                                                                                           |
| name                   | glorot_normal_initializer |         | Initializer name                                                                                                                                             |
| gain                   | 1.4142                    |         | Additional parameter.                                                                                                                                        |
| convs                  |                           |         | Convolution layers. Same parameters as we have in torch.                                                                                                     |
| filters                | 32                        |         | Number of filters.                                                                                                                                           |
| kernel_size            | 8                         |         | Kernel size.                                                                                                                                                 |
| strides                | 4                         |         | Strides                                                                                                                                                      |
| padding                | 0                         |         | Padding                                                                                                                                                      |
| filters                | 64                        |         | Next convolution layer info.                                                                                                                                 |
| kernel_size            | 4                         |         |                                                                                                                                                              |
| strides                | 2                         |         |                                                                                                                                                              |
| padding                | 0                         |         |                                                                                                                                                              |
| filters                | 64                        |         |                                                                                                                                                              |
| kernel_size            | 3                         |         |                                                                                                                                                              |
| strides                | 1                         |         |                                                                                                                                                              |
| padding                | 0                         |         |
| mlp                    |                           |         | MLP Block. Convolution is supported too. See other config examples.                                                                                          |
| units                  |                           |         | Array of sizes of the MLP layers, for example: [512, 256, 128]                                                                                               |
| d2rl                   | False                     |         | Use d2rl architecture from https://arxiv.org/abs/2010.09163.                                                                                                 |
| activation             | elu                       |         | Activations between dense layers.                                                                                                                            |
| initializer            |                           |         | Initializer.                                                                                                                                                 |
| name                   | default                   |         | Initializer name.                                                                                                                                            |
| rnn                    |                           |         | RNN block.                                                                                                                                                   |
| name                   | lstm                      |         | RNN Layer name. lstm and gru are supported.                                                                                                                  |
| units                  | 256                       |         | Number of units.                                                                                                                                             |
| layers                 | 1                         |         | Number of layers                                                                                                                                             |
| before_mlp             | False                     | False   | Apply rnn before mlp block or not.                                                                                                                           |
| config                 |                           |         | RL Config block.                                                                                                                                             |
| reward_shaper          |                           |         | Reward Shaper. Can apply simple transformations.                                                                                                             |
| min_val                | -1                        |         | You can apply min_val, max_val, scale and shift.                                                                                                             |
| scale_value            | 0.1                       | 1       |                                                                                                                                                              |
| normalize_advantage    | True                      | True    | Normalize Advantage.                                                                                                                                         |
| gamma                  | 0.995                     |         | Reward Discount                                                                                                                                              |
| tau                    | 0.95                      |         | Lambda for GAE. Called tau by mistake long time ago because lambda is keyword in python :(                                                                   |
| learning_rate          | 3e-4                      |         | Learning rate.                                                                                                                                               |
| name                   | walker                    |         | Name which will be used in tensorboard.                                                                                                                      |
| save_best_after        | 10                        |         | How many epochs to wait before start saving checkpoint with best score.                                                                                      |
| score_to_win           | 300                       |         | If score is >=value then this value training will stop.                                                                                                      |
| grad_norm              | 1.5                       |         | Grad norm. Applied if truncate_grads is True. Good value is in (1.0, 10.0)                                                                                   |
| entropy_coef           | 0                         |         | Entropy coefficient. Good value for continuous space is 0. For discrete is 0.02                                                                              |
| truncate_grads         | True                      |         | Apply truncate grads or not. It stabilizes training.                                                                                                         |
| env_name               | BipedalWalker-v3          |         | Envinronment name.                                                                                                                                           |
| e_clip                 | 0.2                       |         | clip parameter for ppo loss.                                                                                                                                 |
| clip_value             | False                     |         | Apply clip to the value loss. If you are using normalize_value you don't need it.                                                                            |
| num_actors             | 16                        |         | Number of running actors/environments.                                                                                                                       |
| horizon_length         | 4096                      |         | Horizon length per each actor. Total number of steps will be num_actors*horizon_length * num_agents (if env is not MA num_agents==1).                        |
| minibatch_size         | 8192                      |         | Minibatch size. Total number number of steps must be divisible by minibatch size.                                                                            |
| minibatch_size_per_env | 8                         |         | Minibatch size per env. If specified will overwrite total number number the default minibatch size with minibatch_size_per_env * nume_envs value.            |
| mini_epochs            | 4                         |         | Number of miniepochs. Good value is in [1,10]                                                                                                                |
| critic_coef            | 2                         |         | Critic coef. by default critic_loss = critic_coef * 1/2 * MSE.                                                                                               |
| lr_schedule            | adaptive                  | None    | Scheduler type. Could be None, linear or adaptive. Adaptive is the best for continuous control tasks. Learning rate is changed changed every miniepoch       |
| kl_threshold           | 0.008                     |         | KL threshould for adaptive schedule. if KL < kl_threshold/2 lr = lr * 1.5 and opposite.                                                                      |
| normalize_input        | True                      |         | Apply running mean std for input.                                                                                                                            |
| bounds_loss_coef       | 0.0                       |         | Coefficient to the auxiary loss for continuous space.                                                                                                        |
| max_epochs             | 10000                     |         | Maximum number of epochs to run.                                                                                                                             |
| max_frames             | 5000000                   |         | Maximum number of frames (env steps) to run.                                                                                                                             |
| normalize_value        | True                      |         | Use value running mean std normalization.                                                                                                                    |
| use_diagnostics        | True                      |         | Adds more information into the tensorboard.                                                                                                                  |
| value_bootstrap        | True                      |         | Bootstraping value when episode is finished. Very useful for different locomotion envs.                                                                      |
| bound_loss_type        | regularisation            | None    | Adds aux loss for continuous case. 'regularisation' is the sum of sqaured actions. 'bound' is the sum of actions higher than 1.1.                            |
| bounds_loss_coef       | 0.0005                    | 0       | Regularisation coefficient                                                                                                                                   |
| use_smooth_clamp       | False                     |         | Use smooth clamp instead of regular for cliping                                                                                                              |
| zero_rnn_on_done       | False                     | True    | If False RNN internal state is not reset (set to 0) when an environment is rest. Could improve training in some cases, for example when domain randomization is on |
| player                 |                           |         | Player configuration block.                                                                                                                                  |
| render                 | True                      | False   | Render environment                                                                                                                                           |
| deterministic          | True                      | True    | Use deterministic policy ( argmax or mu) or stochastic.                                                                                                      |
| use_vecenv             | True                      | False   | Use vecenv to create environment for player                                                                                                                  |
| games_num              | 200                       |         | Number of games to run in the player mode.                                                                                                                   |
| env_config             |                           |         | Env configuration block. It goes directly to the environment. This example was take for my atari wrapper.                                                    |
| skip                   | 4                         |         | Number of frames to skip                                                                                                                                     |
| name                   | BreakoutNoFrameskip-v4    |         | The exact name of an (atari) gym env. An example, depends on the training env this parameters can be different.                                                                   |

## Custom network example: 
[simple test network](rl_games/envs/test_network.py)  
This network takes dictionary observation.
To register it you can add code in your __init__.py

```
from rl_games.envs.test_network import TestNetBuilder 
from rl_games.algos_torch import model_builder
model_builder.register_network('testnet', TestNetBuilder)
```
[simple test environment](rl_games/envs/test/rnn_env.py)
[example environment](rl_games/envs/test/example_env.py)  

Additional environment supported properties and functions  

| Field                      | Default Value | Description                                                                                                                                                                                              |
| -------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| use_central_value          | False         | If true than returned obs is expected to be dict with 'obs' and 'state'                                                                                                                                  |
| value_size                 | 1             | Shape of the returned rewards. Network wil support multihead value automatically.                                                                                                                        |
| concat_infos               | False         | Should default vecenv convert list of dicts to the dicts of lists. Very usefull if you want to use value_boostrapping. in this case you need to always return 'time_outs' : True or False, from the env. |
| get_number_of_agents(self) | 1             | Returns number of agents in the environment                                                                                                                                                              |
| has_action_mask(self)      | False         | Returns True if environment has invalid actions mask.                                                                                                                                                    |
| get_action_mask(self)      | None          | Returns action masks if  has_action_mask is true.  Good example is [SMAC Env](rl_games/envs/test/smac_env.py)                                                                                            |


## Release Notes

1.6.1 (Unreleased)
* Fixed Central Value RNN bug which occurs if you train ma multi agent environment.
* Added Deepmind Control PPO benchmark.
* Added a few more experimental ways to train value prediction (OneHot, TwoHot encoding and crossentropy loss instead of L2).
* New methods didn't. It is impossible to turn it on from the yaml files. Once we find an env which trains better it will be added to the config.
* Added shaped reward graph to the tensorboard. 


1.6.0

* Added ONNX export colab example for discrete and continious action spaces. For continuous case LSTM policy example is provided as well.
* Improved RNNs training in continuous space, added option `zero_rnn_on_done`.
* Added NVIDIA CuLE support: https://github.com/NVlabs/cule
* Added player config everride. Vecenv is used for inference.
* Fixed multi-gpu training with central value.
* Fixed max_frames termination condition, and it's interaction with the linear learning rate: https://github.com/Denys88/rl_games/issues/212
* Fixed "deterministic" misspelling issue.
* Fixed Mujoco and Brax SAC configs.
* Fixed multiagent envs statistics reporting. Fixed Starcraft2 SMAC environments.

1.5.2

* Added observation normalization to the SAC.
* Returned back adaptive KL legacy mode.

1.5.1

* Fixed build package issue.

1.5.0

* Added wandb support.
* Added poetry support.
* Fixed various bugs.
* Fixed cnn input was not divided by 255 in case of the dictionary obs.
* Added more envpool mujoco and atari training examples. Some of the results: 15 min Mujoco humanoid training, 2 min atari pong.
* Added Brax and Mujoco colab training examples.
* Added 'seed' command line parameter. Will override seed in config in case it's > 0.
* Deprecated `horovod` in favor of `torch.distributed` ([#171](https://github.com/Denys88/rl_games/pull/171)).

1.4.0

* Added discord channel https://discord.gg/hnYRq7DsQh :)
* Added envpool support with a few atari examples. Works 3-4x time faster than ray.
* Added mujoco results. Much better than openai spinning up ppo results.
* Added tcnn(https://github.com/NVlabs/tiny-cuda-nn) support. Reduces 5-10% of training time in the IsaacGym envs. 
* Various fixes and improvements.

1.3.2

* Added 'sigma' command line parameter. Will override sigma for continuous space in case if fixed_sigma is True.

1.3.1

* Fixed SAC not working

1.3.0

* Simplified rnn implementation. Works a little bit slower but much more stable. 
* Now central value can be non-rnn if policy is rnn.
* Removed load_checkpoint from the yaml file. now --checkpoint works for both train and play.

1.2.0

* Added Swish (SILU) and GELU activations, it can improve Isaac Gym results for some of the envs.
* Removed tensorflow and made initial cleanup of the old/unused code.
* Simplified runner.
* Now networks are created in the algos with load_network method.

1.1.4

* Fixed crash in a play (test) mode in player, when simulation and rl_devices are not the same.
* Fixed variuos multi gpu errors.

1.1.3

* Fixed crash when running single Isaac Gym environment in a play (test) mode.
* Added config parameter ```clip_actions``` for switching off internal action clipping and rescaling

1.1.0

* Added to pypi: ```pip install rl-games```
* Added reporting env (sim) step fps, without policy inference. Improved naming.
* Renames in yaml config for better readability: steps_num to horizon_length amd lr_threshold to kl_threshold



## Troubleshouting

* Some of the supported envs are not installed with setup.py, you need to manually install them
* Starting from rl-games 1.1.0 old yaml configs won't be compatible with the new version: 
    * ```steps_num``` should be changed to ```horizon_length``` amd ```lr_threshold``` to ```kl_threshold```

## Known issues

* Running a single environment with Isaac Gym can cause crash, if it happens switch to at least 2 environments simulated in parallel
    

