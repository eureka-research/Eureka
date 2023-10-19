Release Notes
=============

1.3.4
-----

* Fixed bug when running inferencing on DeXtreme environments.
* Fixed links in examples documentation.
* Minor fixes in documentation.

1.3.3
-----

* Fixed player and bug with AMP training environments.
* Added DeXtreme environments with ADR support.

1.3.2
-----

* Switched all environments that use contacts to use CC_LAST_SUBSTEP collection mode to avoid bug with CC_ALL_SUBSTEP mode. The CC_ALL_SUBSTEP mode can produce incorrect contact forces. Only HumanoidAMP and Factory environments are affected by this. 
* Added SAC training examples for Ant and Humanoid envs. To run: ``python train.py task=AntSAC train=AntSAC`` and ``python train.py task=HumanoidSAC train=HumanoidSAC``
* Fix shadow hand and allegro hand random joint position sampling on reset.
* Switched to using IsaacAlgoObserver from rl_games instead of the custom RLGPUAlgoObserver.

1.3.1
-----

* Moved domain randomization utility code into IsaacGymEnvs.
* Tweaks and additional documentation for Factory examples and SDF collisions.

1.3.0
-----

* Added Factory Environments demonstrating RL with SDF collisions.
* Added Franka Cube Stacking task. Can use Operational Space Control (OSC) or joint torque control.
* Added support for [WandB](https://wandb.ai/) via adding `wandb_activate=True` on the training command line.
* Improved handling of episode timeouts (`self.timeout_buf`, see 1.1.0) which might have caused training issues for 
configurations with `value_bootstrap: True`. This fix results in slightly faster training on Ant & Humanoid locomotion tasks.
* Added retargeting data for SFU Motion Capture Database.
* Deprecated `horovod` in favor of `torch.distributed` for better performance in multi-GPU settings.
* Added an environment creation API `isaacgymenvs.make(task_name)` which creates a vectorized environment compatible with 3rd party RL libraries. 
* Added a utility to help capture the videos of the agent's gameplay via `python train.py capture_video=True` which creates a `videos` folder.
* Fixed an issue with Anymal Terrain environment resets.
* Improved allegro.urdf which now includes more precise collision shapes and masses/inertias of finger links.
* Added a pre-commit utility to identify incorrect spelling.

1.2.0
-----

* Added AMP (Adversarial Motion Priors) training environment.
* Minor changes in base VecTask class.

1.1.0
-----

* Added Anymal Rough Terrain and Trifinger training environments.
* Added `self.timeout_buf` that stores the information if the reset happened because of the episode reached to the maximum length or because of some other termination conditions. Is stored in extra info: `self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)`.  Updated PPO configs to use this information during training with `value_bootstrap: True`.

1.0.0
-----

* Initial release
