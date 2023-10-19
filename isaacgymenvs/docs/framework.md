RL Framework
===================

Overview
--------

Our training examples run using a third-party highly-optimized RL library,
[rl_games](https://github.com/Denys88/rl_games). This also demonstrates
how our framework can be used with other RL libraries.

RL Games will be installed automatically along with `isaacgymenvs`.
Otherwise, to install **rl_games** manually the following instructions should be performed:

```bash
git clone https://github.com/Denys88/rl_games.git
pip install -e .
```

For all the sample tasks provided, we include training configurations
for rl_games, denoted with the suffixes `*PPO.yaml`.
These files are located in `isaacgymenvs/config/train`.
The appropriate config file will be selected
automatically based on the task being executed and the script that it is
being launched from. To launch a task using rl-games, run
`python train.py`.

For a list of the sample tasks we provide, refer to the
[RL List of Examples](rl.md)

Class Definition
----------------

The base class for Isaac Gym's RL framework is `VecTask` in [vec_task.py](../isaacgymenvs/tasks/base/vec_task.py).

The `VecTask` class is designed to act as a parent class for all RL tasks
using Isaac Gym's RL framework. It provides an interface for interaction
with RL algorithms and includes functionalities that are required for
all RL tasks.


The `VecTask` constructor takes a configuration dictionary containing numerous parameters required:

`device_type` - the type of device used for simulation. `cuda` or `cpu`.

`device_id` - ID of the device used for simulation. eg `0` for a single GPU workstation.

`rl_device` - Full `name:id` string of the device that the RL framework is using.

`headless` - `True`/`False` depending on whether you want the simulation to run the simulation with a viewer.

`physics_engine` - which physics engine to use. Must be `"physx"` or `"flex"`.

`env` - a dictionary with environment-specific parameters.
Can include anything in here you want depending on the specific parameters, but key ones which you must provide are:
* `numEnvs` - number of environments being simulated in parallel
* `numObservations` - size of the observation vector used for each environment.
* `numActions` - size of the actions vector.

Other optional parameters are 
* `numAgents` - for multi-agent environments. Defaults to `1`
* `numStates` - for size of state vector for training with asymmetric actor-critic. 
* `controlFrequencyInv` - control decimation, ie. how many simulator steps between RL actions. Defaults to 1.
* `clipObservations` - range to clip observations to. Defaults to `inf` (+-infinity).
* `clipActions` - range to clip actions to. Defaults to `1` (+-1).
* `enableCameraSensors` - set to `True` if camera sensors are used in the environment.


The `__init__` function of `VecTask` triggers a call to `create_sim()`,
which must be implemented by the extended classes. 
It will then initialize buffers required for RL on the device specified. These include observation buffer, reward
buffer, reset buffer, progress buffer, randomization buffer, and an optional extras array for passing in any additional
information to the RL algorithm.

A call to `prepare_sim()` will also be made to initialize the internal data
structures for simulation. `set_viewer()` is also called, which, if running with a viewer,
this function will also initialize the viewer and create keyboard shortcuts for quitting
the application (ESC) and disabling/enabling rendering (V).

The `step` function is designed to guide the workflow of each RL
iteration. This function can be viewed in three parts:
`pre_physics_step`, `simulate`, and `post_physics_step`.
`pre_physics_step` should be implemented to perform any computations
required before stepping the physics simulation. As an example, applying
actions from the policy should happen in `pre_physics_step`. `simulate`
is then called to step the physics simulation. `post_physics_step`
should implement computations performed after stepping the physics
simulation, e.g. computing rewards and observations.

`VecTask` also provides an implementation of `render` to step graphics if
a viewer is initialized.

Additionally, VecTask provides an interface to perform Domain
Randomization via the `apply_randomizations` method. For more details,
please see [Domain Randomization](domain_randomization.md).


Creating a New Task
-------------------

Creating a new task is straight-forward using Isaac Gym's RL framework.
The first step is to create a new script file in [isaacgymenvs/tasks](../isaacgymenvs/tasks).

To use Isaac Gym's APIs, we need the following imports

```python
from isaacgym import gymtorch
from isaacgym import gymapi

from .base.vec_task import VecTask
```

Then, we need to create a Task class that extends from VecTask

```python
class MyNewTask(VecTask):
```

The `__init__` method should take 3 arguments: a config dict conforming to the
specifications described above (this will be generated from hydra config), `sim_device`, the device string representing
where the simulation will be run, and `headless`, which specifies whether or not to run in headless mode.

In the `__init__` method of MyNewTask, make sure to make a call to
`VecTask`'s `__init__` to initialize the simulation, providing the
config dictionary with members as described above:

```python
super().__init__(
    cfg=config_dict
)
```

Then, we can initialize state tensors that we may need for our task. For
example, we can initialize the DOF state tensor

```python
dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
```

There are a few methods that must be implemented by a child class of
VecTask: `create_sim`, `pre_physics_step`, `post_physics_step`.

```python
def create_sim(self):
    # implement sim set up and environment creation here
    #    - set up-axis
    #    - call super().create_sim with device args (see docstring)
    #    - create ground plane
    #    - set up environments

def pre_physics_step(self, actions):
    # implement pre-physics simulation code here
    #    - e.g. apply actions

def post_physics_step(self):
    # implement post-physics simulation code here
    #    - e.g. compute reward, compute observations
```

To launch the new task from `train.py`, add your new
task to the imports and `isaacgym_task_map` dict in the `tasks` [\_\_init\_\_.py file](../isaacgymenvs/tasks/__init__.py).


```python
from isaacgymenvs.tasks.my_new_task import MyNewTask
...
isaac_gym_task_map = {
    'Anymal': Anymal,
    # ...
    'MyNewTask': MyNewTask,
}
```

You will also need to create config files for task and training, which will be passed in dictionary form to the first
`config` argument of your task. The `task` config, which goes in the [corresponding config folder](../isaacgymenvs/cfg/task)
must have a `name` in the root matching the task name you put in the `isaac_gym_task_map` above. You should name your
task config the same as in the Isaac Gym task map, eg. `Anymal` becomes [`Anymal.yaml`](../isaacgymenvs/cfg/task/Anymal.yaml).

You also need a `train` config specifying RL Games arguments. This should go in the [corresponding config folder](../isaacgymenvs/cfg/train).
The file should have the postfix `PPO`, ie `Anymal` becomes [`AnymalPPO.yaml`](../isaacgymenvs/cfg/train/AnymalPPO.yaml).

Then, you can run your task with `python train.py task=MyNewTask`.


Updating an Existing Environment
--------------------------------

If you have existing environments set up with Isaac Gym Preview 2 release or earlier, it is simple to convert your tasks to the new RL framework in IsaacGymEnvs. Here are a few pointers to help you get started.

### Imports ###
* The `torch_jit_utils` script has been moved to IsaacGymEnvs. Tasks that are importing from `rlgpu.utils.torch_jit_utils` should now import from `utils.torch_jit_utils`.
* The original `BaseTask` class has been converted to `VecTask` in IsaacGymEnvs. All tasks inheriting from the previous `BaseTask` should modify `from rlgpu.tasks.base.base_task import BaseTask` to `from .base.vec_task import VecTask`.

### Class Definition ###
* Your task class should now inherit from `VecTask` instead of the previous `BaseTask`.
* Arguments required for class initialization has been simplified. The task `__init__()` method now only requires `cfg`, `sim_device`, and `headless` as arguments.
* It is no longer required to set `self.sim_params` and `self.physics_engine` in the `__init__()` method of your task definition.
* Making a call to `VecTask`'s `__init__()` method requires 3 more arguments: `rl_device`, `sim_device` and `headless`. As an example, modify the line of code to `super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, headless=headless)`.
* `VecTask` now defines a `reset_idx()` function that should be implemented in an environment class. It resets environments with the provided indices.
* Note that `VecTask` now defines a `reset()` method that does not accept environment indices as arguments. To avoid naming conflicts, consider renaming the `reset()` method inside your task definition.

### Asset Loading ###
* Assets have been moved to IsaacGymEnvs (with some still remaining in IsaacGym for use in examples). Please make sure the paths to your assets remain valid in the new IsaacGymEnvs setup.
* Assets are now located under `assets/`.

### Configs ###
* Some config parameters are now updated to work with resolvers and Hydra. Please refer to an example config in `cfg/` for details.
* For task configs, the following are modified: `physics_engine`, `numEnvs`, `use_gpu_pipeline`, `num_threads`, `solver_type`, `use_gpu`, `num_subscenes`.
* For train configs, the following are modified: `seed`, `load_checkpoint`, `load_path`, `name`, `full_experiment_name`, `num_actors`, `max_epochs`.
* Also note a few naming changes required for the latest version of rl_games: `lr_threshold` --> `kl_threshold`, `steps_num` --> `horizon_length`.
