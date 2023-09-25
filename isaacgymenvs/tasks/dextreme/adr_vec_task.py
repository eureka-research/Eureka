# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch
import numpy as np
import operator, random
from copy import deepcopy
from isaacgymenvs.utils.utils import nested_dict_get_attr, nested_dict_set_attr

from collections import deque
from enum import Enum


import sys

import abc
from abc import ABC

from omegaconf import ListConfig


class RolloutWorkerModes:
    ADR_ROLLOUT = 0 # rollout with current ADR params
    ADR_BOUNDARY = 1 # rollout with params on boundaries of ADR, used to decide whether to expand ranges
    TEST_ENV = 2 # rollout wit default DR params, used to measure overall success rate. (currently unused)

from isaacgymenvs.tasks.base.vec_task import Env, VecTask


class EnvDextreme(Env):

    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool, use_dict_obs: bool):
        
        Env.__init__(self, config, rl_device, sim_device, graphics_device_id, headless)

        self.use_dict_obs = use_dict_obs

        if self.use_dict_obs:
        
            self.obs_dims = config["env"]["obsDims"]
            self.obs_space = spaces.Dict(
                {
                    k: spaces.Box(
                        np.ones(shape=dims) * -np.Inf, np.ones(shape=dims) * np.Inf
                    )
                    for k, dims in self.obs_dims.items()
                }
            )

        else:
        
            self.num_observations = config["env"]["numObservations"]
            self.num_states = config["env"].get("numStates", 0)

            self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
    
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass


class VecTaskDextreme(EnvDextreme, VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs=False):        
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        
        EnvDextreme.__init__(self, config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs=use_dict_obs)

        self.sim_params = self._VecTask__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.virtual_display = None 

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True

        self.randomize = self.cfg["task"]["randomize"]
        self.randomize_obs_builtin = "observations" in self.cfg["task"].get("randomization_params", {})
        self.randomize_act_builtin = "actions" in self.cfg["task"].get("randomization_params", {})

        self.randomized_suffix = "randomized"

        if self.use_dict_obs and self.randomize and self.randomize_obs_builtin:
            self.randomisation_obs = set(self.obs_space.keys()).intersection(set(self.randomization_params['observations'].keys()))
            for obs_name in self.randomisation_obs:
                self.obs_space[f"{obs_name}_{self.randomized_suffix}"] = self.obs_space[obs_name]
                self.obs_dims[f"{obs_name}_{self.randomized_suffix}"] = self.obs_dims[obs_name]
                
            self.obs_randomizations = {}
        elif self.randomize_obs_builtin:
            self.obs_randomizations = None

        self.action_randomizations = None

        self.original_props = {}

        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers

        if self.use_dict_obs:
            self.obs_dict = {
                k: torch.zeros(
                    (self.num_envs, *dims), device=self.device, dtype=torch.float
                )
                for k, dims in self.obs_dims.items()
            }
            print("Obs dictinary: ")
            print(self.obs_dims)
            # print(self.obs_dict)
            for k, dims in self.obs_dims.items():
                print("1")
                print(dims)

            self.obs_dict_repeat = {
                
                 k: torch.zeros(
                    (self.num_envs, *dims), device=self.device, dtype=torch.float
                )
                for k, dims in self.obs_dims.items()
            }
        else:
            self.obs_dict = {}
            self.obs_buf = torch.zeros(
                (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
            self.states_buf = torch.zeros(
                (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the priviledged observations for asymmetric training)."""
        if self.use_dict_obs:
            raise NotImplementedError("No states in vec task when `use_dict_obs=True`")
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""


    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.action_randomizations is not None and self.randomize_act_builtin:
            actions = self.action_randomizations['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        # cannot randomise in the env because of missing suffix in the observation dict
        if self.randomize and self.randomize_obs_builtin and self.use_dict_obs and len(self.obs_randomizations) > 0:
            for obs_name, v in self.obs_randomizations.items():

                self.obs_dict[f"{obs_name}_{self.randomized_suffix}"] = v['noise_lambda'](self.obs_dict[obs_name])

                # Random cube pose 
                if hasattr(self, 'enable_random_obs') and self.enable_random_obs and obs_name == 'object_pose_cam':
                    self.obs_dict[f"{obs_name}_{self.randomized_suffix}"] \
                        = self.get_random_cube_observation(self.obs_dict[f"{obs_name}_{self.randomized_suffix}"])

            if hasattr(self, 'enable_random_obs') and self.enable_random_obs:

                relative_rot = self.get_relative_rot(self.obs_dict['object_pose_cam_'+ self.randomized_suffix][:, 3:7],
                                                            self.obs_dict['goal_pose'][:, 3:7])

                v = self.obs_randomizations['goal_relative_rot_cam']
                self.obs_dict["goal_relative_rot_cam_" + self.randomized_suffix] = v['noise_lambda'](relative_rot)

        elif self.randomize and self.randomize_obs_builtin and not self.use_dict_obs and self.obs_randomizations is not None:
            self.obs_buf = self.obs_randomizations['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        if self.use_dict_obs:
            obs_dict_ret = {
                k: torch.clone(torch.clamp(t, -self.clip_obs, self.clip_obs)).to(
                    self.rl_device
                )
                for k, t in self.obs_dict.items()
            }

            return obs_dict_ret, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

            return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        if self.use_dict_obs:
            obs_dict_ret = {
                k: torch.clone(
                    torch.clamp(t, -self.clip_obs, self.clip_obs).to(self.rl_device)
                )
                for k, t in self.obs_dict.items()
            }

            return obs_dict_ret
        else:

            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

            return self.obs_dict

    """
    Domain Randomization methods
    """

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """

        if self.use_adr:
            return dict(adr_params=self.adr_params)
        else:
            return {}

    def set_env_state(self, env_state):

        if env_state is None:
            return

        for key in self.get_env_state().keys():

            if key == "adr_params" and self.use_adr and not self.adr_load_from_checkpoint:
                print("Skipping loading ADR params from checkpoint...")
                continue
            value = env_state.get(key, None)
            if value is None:
                continue

            self.__dict__[key] = value
            print(f'Loaded env state value {key}:{value}')

        if self.use_adr:
            print(f'ADR Params after loading from checkpoint: {self.adr_params}')

                  
    def get_randomization_dict(self, dr_params, obs_shape):
        dist = dr_params["distribution"]
        op_type = dr_params["operation"]
        sched_type = dr_params["schedule"] if "schedule" in dr_params else None
        sched_step = dr_params["schedule_steps"] if "schedule" in dr_params else None
        op = operator.add if op_type == 'additive' else operator.mul

        if not self.use_adr:
            apply_white_noise_prob = dr_params.get("apply_white_noise", 0.5)

        if sched_type == 'linear':
            sched_scaling = 1.0 / sched_step * \
                min(self.last_step, sched_step)
        elif sched_type == 'constant':
            sched_scaling = 0 if self.last_step < sched_step else 1
        else:
            sched_scaling = 1

        if dist == 'gaussian':
            mu, var = dr_params["range"]
            mu_corr, var_corr = dr_params.get("range_correlated", [0., 0.])

            if op_type == 'additive':
                mu *= sched_scaling
                var *= sched_scaling
                mu_corr *= sched_scaling
                var_corr *= sched_scaling
            elif op_type == 'scaling':
                var = var * sched_scaling  # scale up var over time
                mu = mu * sched_scaling + 1.0 * \
                    (1.0 - sched_scaling)  # linearly interpolate

                var_corr = var_corr * sched_scaling  # scale up var over time
                mu_corr = mu_corr * sched_scaling + 1.0 * \
                    (1.0 - sched_scaling)  # linearly interpolate
            
            local_params = {
                'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr,
                'corr': torch.randn(self.num_envs, *obs_shape, device=self.device)            
            }
            if not self.use_adr:
                local_params['apply_white_noise_mask'] = (torch.rand(self.num_envs, device=self.device) < apply_white_noise_prob).float()
            def noise_lambda(tensor, params=local_params):
                corr = local_params['corr']
                corr = corr * params['var_corr'] + params['mu_corr']
                if self.use_adr:
                    return op(
                        tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])
                else:
                    return op(
                    tensor, corr + torch.randn_like(tensor) * params['apply_white_noise_mask'].view(-1, 1) * params['var'] + params['mu'])


        elif dist == 'uniform':
            lo, hi = dr_params["range"]
            lo_corr, hi_corr = dr_params.get("range_correlated", [0., 0.])

            if op_type == 'additive':
                lo *= sched_scaling
                hi *= sched_scaling
                lo_corr *= sched_scaling
                hi_corr *= sched_scaling
            elif op_type == 'scaling':
                lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

            local_params = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr,
                            'corr': torch.rand(self.num_envs, *obs_shape, device=self.device)
                            }
            if not self.use_adr:
                local_params['apply_white_noise_mask'] = (torch.rand(self.num_envs, device=self.device) < apply_white_noise_prob).float()

            def noise_lambda(tensor, params=local_params):
                corr = params['corr']
                corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                if self.use_adr:
                    return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])
                else:
                    return op(tensor, corr + torch.rand_like(tensor) * params['apply_white_noise_mask'].view(-1, 1) * (params['hi'] - params['lo']) + params['lo'])


        else:
            raise NotImplementedError

        # return {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}
        return {'noise_lambda': noise_lambda, 'corr_val': local_params['corr']}

class ADRVecTask(VecTaskDextreme):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs=False):

        self.adr_cfg = self.cfg["task"].get("adr", {})
        self.use_adr = self.adr_cfg.get("use_adr", False)

        self.all_env_ids = torch.tensor(list(range(self.cfg["env"]["numEnvs"])), dtype=torch.long, device=sim_device)

        if self.use_adr:

            self.worker_adr_boundary_fraction = self.adr_cfg["worker_adr_boundary_fraction"]
            self.adr_queue_threshold_length = self.adr_cfg["adr_queue_threshold_length"]

            self.adr_objective_threshold_low = self.adr_cfg["adr_objective_threshold_low"]
            self.adr_objective_threshold_high = self.adr_cfg["adr_objective_threshold_high"]

            self.adr_extended_boundary_sample = self.adr_cfg["adr_extended_boundary_sample"]

            self.adr_rollout_perf_alpha = self.adr_cfg["adr_rollout_perf_alpha"]

            self.update_adr_ranges = self.adr_cfg["update_adr_ranges"]

            self.adr_clear_other_queues = self.adr_cfg["clear_other_queues"]

            self.adr_rollout_perf_last = None

            self.adr_load_from_checkpoint = self.adr_cfg["adr_load_from_checkpoint"]

            assert self.randomize, "Worker mode currently only supported when Domain Randomization is turned on"

            # 0 = rollout worker
            # 1 = ADR worker (see https://arxiv.org/pdf/1910.07113.pdf Section 5)
            # 2 = eval worker
            # rollout type is selected when an environment gets randomized
            self.worker_types = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.long, device=sim_device)

            self.adr_tensor_values = {}

            self.adr_params = self.adr_cfg["params"]

            self.adr_params_keys = list(self.adr_params.keys())
            # list of params which rely on patching the built in domain randomisation
            self.adr_params_builtin_keys = []
            
            for k in self.adr_params:
                self.adr_params[k]["range"] = self.adr_params[k]["init_range"]
                if "limits" not in self.adr_params[k]:
                    self.adr_params[k]["limits"] = [None, None]
                if "delta_style" in self.adr_params[k]:
                    assert self.adr_params[k]["delta_style"] in ["additive", "multiplicative"]
                else:
                    self.adr_params[k]["delta_style"] = "additive"
                
                if "range_path" in self.adr_params[k]:
                    self.adr_params_builtin_keys.append(k)
                else: # normal tensorised ADR param
                    param_type = self.adr_params[k].get("type", "uniform")
                    dtype = torch.long if param_type == "categorical" else torch.float
                    self.adr_tensor_values[k] = torch.zeros(self.cfg["env"]["numEnvs"], device=sim_device, dtype=dtype)

            self.num_adr_params = len(self.adr_params)

            # modes for ADR workers. 
            # there are 2n modes, where mode 2n is lower range and mode 2n+1 is upper range for DR parameter n
            self.adr_modes = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.long, device=sim_device)

            self.adr_objective_queues = [deque(maxlen=self.adr_queue_threshold_length) for _ in range(2*self.num_adr_params)]

        super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs=use_dict_obs)


    def get_current_adr_params(self, dr_params):
        """Splices the current ADR parameters into the requried ranges"""

        current_adr_params = copy.deepcopy(dr_params)
        
        for k in self.adr_params_builtin_keys:
            nested_dict_set_attr(current_adr_params, self.adr_params[k]["range_path"], self.adr_params[k]["range"])
        
        return current_adr_params
    def get_dr_params_by_env_id(self, env_id, default_dr_params, current_adr_params):
        """Returns the (dictionary) DR params for a particular env ID.
        (only applies to env randomisations, for tensor randomisations see `sample_adr_tensor`.)

        Params:
            env_id: which env ID to get the dict for.
            default_dr_params: environment default DR params.
            current_adr_params: current dictionary of DR params with current ADR ranges patched in.
        Returns:
            a patched dictionary with the env randomisations corresponding to the env ID.
        """

        env_type = self.worker_types[env_id]
        if env_type == RolloutWorkerModes.ADR_ROLLOUT: # rollout worker, uses current ADR params
            return current_adr_params
        elif env_type == RolloutWorkerModes.ADR_BOUNDARY: # ADR worker, substitute upper or lower bound as entire range for this env
            adr_mode = int(self.adr_modes[env_id])

            env_adr_params = copy.deepcopy(current_adr_params)
            adr_id = adr_mode // 2 # which adr parameter
            adr_bound = adr_mode % 2 # 0 = lower, 1 = upper
            param_name = self.adr_params_keys[adr_id]
            
            # this DR parameter is randomised as a tensor not through normal DR api
            # if not "range_path" in self.adr_params[self.adr_params_keys[adr_id]]:
            if not param_name in self.adr_params_builtin_keys:
                return env_adr_params
            
            if self.adr_extended_boundary_sample:
                boundary_value = self.adr_params[param_name]["next_limits"][adr_bound] 
            else:
                boundary_value = self.adr_params[param_name]["range"][adr_bound]
            new_range = [boundary_value, boundary_value]
            
            nested_dict_set_attr(env_adr_params, self.adr_params[param_name]["range_path"], new_range)

            return env_adr_params
        elif env_type == RolloutWorkerModes.TEST_ENV:  # eval worker, uses default fixed params
            return default_dr_params
        else:
            raise NotImplementedError
    
    def modify_adr_param(self, param, direction, adr_param_dict, param_limit=None):
        """Modify an ADR param.
        
        Args:
            param: current value of the param.
            direction: what direction to move the ADR parameter ('up' or 'down')
            adr_param_dict: dictionary of ADR parameter, used to read delta and method of applying delta
            param_limit: limit of the parameter (upper bound for 'up' and lower bound for 'down' mode)
        Returns:
            whether the param was updated
        """
        op = adr_param_dict["delta_style"]
        delta = adr_param_dict["delta"]
        
        if direction == 'up':

            if op == "additive":
                new_val = param + delta
            elif op == "multiplicative":
                assert delta > 1.0, "Must have delta>1 for multiplicative ADR update."
                new_val = param * delta
            else:
                raise NotImplementedError

            if param_limit is not None:
                new_val = min(new_val, param_limit)
            
            changed = abs(new_val - param) > 1e-9
            
            return new_val, changed

        elif direction == 'down':

            if op == "additive":
                new_val = param - delta
            elif op == "multiplicative":
                assert delta > 1.0, "Must have delta>1 for multiplicative ADR update."
                new_val = param / delta
            else:
                raise NotImplementedError

            if param_limit is not None:
                new_val = max(new_val, param_limit)
            
            changed = abs(new_val - param) > 1e-9
            
            return new_val, changed
        else:
            raise NotImplementedError
    
    @staticmethod
    def env_ids_from_mask(mask):
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)
    
    def sample_adr_tensor(self, param_name, env_ids=None):
        """Samples the values for a particular ADR parameter as a tensor.
        Sets the value as a side-effect in the dictionary of current adr tensors.

        Args:
            param_name: name of the parameter to sample
            env_ids: env ids to sample
        Returns:
            (len(env_ids), tensor_dim) tensor of sampled parameter values,
            where tensor_dim is the trailing dimension of the generated tensor as
            specifide in the ADR conifg
        
        """

        if env_ids is None:
            env_ids = self.all_env_ids

        sample_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sample_mask[env_ids] = True
        
        params = self.adr_params[param_name]
        param_range = params["range"]
        next_limits = params.get("next_limits", None)
        param_type = params.get("type", "uniform")

        n = self.adr_params_keys.index(param_name)

        low_idx = 2*n
        high_idx = 2*n + 1

        adr_workers_low_mask = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == low_idx) & sample_mask
        adr_workers_high_mask = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == high_idx) & sample_mask

        rollout_workers_mask = (~adr_workers_low_mask) & (~adr_workers_high_mask) & sample_mask
        rollout_workers_env_ids = self.env_ids_from_mask(rollout_workers_mask)


        if param_type == "uniform":
            result = torch.zeros((len(env_ids),), device=self.device, dtype=torch.float)

            uniform_noise_rollout_workers = \
                torch.rand((rollout_workers_env_ids.shape[0],), device=self.device, dtype=torch.float) \
                * (param_range[1] - param_range[0]) + param_range[0]
            
            result[rollout_workers_mask[env_ids]] = uniform_noise_rollout_workers
            if self.adr_extended_boundary_sample:
                result[adr_workers_low_mask[env_ids]] = next_limits[0]
                result[adr_workers_high_mask[env_ids]] = next_limits[1]
            else:
                result[adr_workers_low_mask[env_ids]] = param_range[0]
                result[adr_workers_high_mask[env_ids]] = param_range[1]
        elif param_type == "categorical":
            result = torch.zeros((len(env_ids), ), device=self.device, dtype=torch.long)

            uniform_noise_rollout_workers = torch.randint(int(param_range[0]), int(param_range[1])+1, size=(rollout_workers_env_ids.shape[0], ), device=self.device)
            
            result[rollout_workers_mask[env_ids]] = uniform_noise_rollout_workers
            result[adr_workers_low_mask[env_ids]] = int(next_limits[0] if self.adr_extended_boundary_sample else param_range[0])
            result[adr_workers_high_mask[env_ids]] = int(next_limits[1] if self.adr_extended_boundary_sample else param_range[1])
        else:
            raise NotImplementedError(f"Unknown distribution type {param_type}")
        
        self.adr_tensor_values[param_name][env_ids] = result

        return result
    
    def get_adr_tensor(self, param_name, env_ids=None):
        """Returns the current value of an ADR tensor.
        """
        if env_ids is None:
            return self.adr_tensor_values[param_name]
        else:
            return self.adr_tensor_values[param_name][env_ids]
    
    def recycle_envs(self, recycle_envs):
        """Recycle the workers that have finished their episodes or to be reassigned etc.

        Args:
            recycle_envs: env_ids of environments to be recycled
        
        """
        worker_types_rand = torch.rand(len(recycle_envs), device=self.device, dtype=torch.float)

        new_worker_types = torch.zeros(len(recycle_envs), device=self.device, dtype=torch.long)
        
        # Choose new types for wokrers 
        new_worker_types[(worker_types_rand < self.worker_adr_boundary_fraction)] = RolloutWorkerModes.ADR_ROLLOUT
        new_worker_types[(worker_types_rand >= self.worker_adr_boundary_fraction)] = RolloutWorkerModes.ADR_BOUNDARY

        self.worker_types[recycle_envs] = new_worker_types
        
        # resample the ADR modes (which boundary values to sample) for the given environments (only applies to ADR_BOUNDARY mode)
        self.adr_modes[recycle_envs] = torch.randint(0, self.num_adr_params * 2, (len(recycle_envs),), dtype=torch.long, device=self.device)
    
    def adr_update(self, rand_envs, adr_objective):
        """Performs ADR update step (implements algorithm 1 from https://arxiv.org/pdf/1910.07113.pdf).
        """

        rand_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        rand_env_mask[rand_envs] = True

        total_nats = 0.0 # measuring entropy

        if self.update_adr_ranges:

            adr_params_iter = list(enumerate(self.adr_params))
            random.shuffle(adr_params_iter)
            
            # only recycle once
            already_recycled = False

            for n, adr_param_name in adr_params_iter:
                
                # mode index for environments evaluating lower ADR bound
                low_idx = 2*n
                # mode index for environments evaluating upper ADR bound
                high_idx = 2*n+1

                adr_workers_low = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == low_idx)
                adr_workers_high = (self.worker_types == RolloutWorkerModes.ADR_BOUNDARY) & (self.adr_modes == high_idx)
                
                # environments which will be evaluated for ADR (finished the episode) and which are evaluating performance at the
                # lower and upper boundaries
                adr_done_low = rand_env_mask & adr_workers_low
                adr_done_high = rand_env_mask & adr_workers_high
                
                # objective value at environments which have been evaluating the lower bound of ADR param n 
                objective_low_bounds = adr_objective[adr_done_low]
                # objective value at environments which have been evaluating the upper bound of ADR param n 
                objective_high_bounds = adr_objective[adr_done_high]

                # add the success of objectives to queues
                self.adr_objective_queues[low_idx].extend(objective_low_bounds.cpu().numpy().tolist())
                self.adr_objective_queues[high_idx].extend(objective_high_bounds.cpu().numpy().tolist())

                low_queue = self.adr_objective_queues[low_idx]
                high_queue = self.adr_objective_queues[high_idx]

                mean_low = np.mean(low_queue) if len(low_queue) > 0 else 0.
                mean_high = np.mean(high_queue) if len(high_queue) > 0 else 0.

                current_range = self.adr_params[adr_param_name]["range"]
                range_lower = current_range[0]
                range_upper = current_range[1]

                range_limits = self.adr_params[adr_param_name]["limits"]
                init_range = self.adr_params[adr_param_name]["init_range"]

                # one step beyond the current ADR values
                [next_limit_lower, next_limit_upper] = self.adr_params[adr_param_name].get("next_limits", [None, None])
                
                changed_low, changed_high = False, False
                
                if len(low_queue) >= self.adr_queue_threshold_length:

                    changed_low = False

                    if mean_low < self.adr_objective_threshold_low:
                        # increase lower bound
                        range_lower, changed_low = self.modify_adr_param(
                            range_lower, 'up', self.adr_params[adr_param_name], param_limit=init_range[0]
                        )

                    elif mean_low > self.adr_objective_threshold_high:
                        # reduce lower bound
                        range_lower, changed_low = self.modify_adr_param(
                            range_lower, 'down', self.adr_params[adr_param_name], param_limit=range_limits[0]
                        )
                        
                    # if the ADR boundary is changed, workers working from the old paremeters become invalid.
                    # Therefore, while we use the data from them to train, we can no longer use them to evaluate DR at the boundary
                    if changed_low:
                        print(f'Changing {adr_param_name} lower bound. Queue length {len(self.adr_objective_queues[low_idx])}. Mean perf: {mean_low}. Old val: {current_range[0]}. New val: {range_lower}')
                        self.adr_objective_queues[low_idx].clear()
                        self.worker_types[adr_workers_low] = RolloutWorkerModes.ADR_ROLLOUT
                
                if len(high_queue) >= self.adr_queue_threshold_length:

                    if mean_high < self.adr_objective_threshold_low:
                        # reduce upper bound
                        range_upper, changed_high = self.modify_adr_param(
                            range_upper, 'down', self.adr_params[adr_param_name], param_limit=init_range[1]
                        )
                    elif mean_high > self.adr_objective_threshold_high:
                        # increase upper bound
                        range_upper, changed_high = self.modify_adr_param(
                            range_upper, 'up', self.adr_params[adr_param_name], param_limit=range_limits[1]
                        )
                
                    # if the ADR boundary is changed, workers working from the old paremeters become invalid.
                    # Therefore, while we use the data from them to train, we can no longer use them to evaluate DR at the boundary
                    if changed_high:
                        print(f'Changing upper bound {adr_param_name}. Queue length {len(self.adr_objective_queues[high_idx])}. Mean perf {mean_high}. Old val: {current_range[1]}. New val: {range_upper}')
                        self.adr_objective_queues[high_idx].clear()
                        self.worker_types[adr_workers_high] = RolloutWorkerModes.ADR_ROLLOUT

                if changed_low or next_limit_lower is None:
                    next_limit_lower, _ = self.modify_adr_param(range_lower, 'down', self.adr_params[adr_param_name], param_limit=range_limits[0])

                if changed_high or next_limit_upper is None:
                    next_limit_upper, _ = self.modify_adr_param(range_upper, 'up', self.adr_params[adr_param_name], param_limit=range_limits[1])

                self.adr_params[adr_param_name]["range"] = [range_lower, range_upper]

                if not self.adr_params[adr_param_name]["delta"] < 1e-9: # disabled
                    upper_lower_delta = range_upper - range_lower

                    if upper_lower_delta < 1e-3:
                        upper_lower_delta = 1e-3

                    nats = np.log(upper_lower_delta)
                    total_nats += nats
                    # print(f'nats {nats} delta {upper_lower_delta} range lower {range_lower} range upper {range_upper}')

                self.adr_params[adr_param_name]["next_limits"] = [next_limit_lower, next_limit_upper]

                if hasattr(self, 'extras') and ((changed_high or changed_low) or self.last_step % 100 == 0): # only log so often to prevent huge log files with ADR vars
                    self.extras[f'adr/params/{adr_param_name}/lower'] = range_lower
                    self.extras[f'adr/params/{adr_param_name}/upper'] = range_upper
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/lower/value'] = mean_low
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/lower/queue_len'] = len(low_queue)
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/upper/value'] = mean_high
                    self.extras[f'adr/objective_perf/boundary/{adr_param_name}/upper/queue_len'] = len(high_queue)
                
                if self.adr_clear_other_queues and (changed_low or changed_high):

                    for q in self.adr_objective_queues:
                        q.clear()
                    recycle_envs = torch.nonzero((self.worker_types == RolloutWorkerModes.ADR_BOUNDARY), as_tuple=False).squeeze(-1)
                    self.recycle_envs(recycle_envs)
                    already_recycled = True
                    break
            
            if hasattr(self, 'extras') and self.last_step % 100 == 0: # only log so often to prevent huge log files with ADR vars
                mean_perf = adr_objective[rand_env_mask & (self.worker_types == RolloutWorkerModes.ADR_ROLLOUT)].mean()
                if self.adr_rollout_perf_last is None:
                    self.adr_rollout_perf_last = mean_perf
                else:
                    self.adr_rollout_perf_last = self.adr_rollout_perf_last * self.adr_rollout_perf_alpha + mean_perf * (1-self.adr_rollout_perf_alpha)
                self.extras[f'adr/objective_perf/rollouts'] = self.adr_rollout_perf_last
                self.extras[f'adr/npd'] = total_nats / len(self.adr_params)
            
            if not already_recycled: 
                self.recycle_envs(rand_envs)
        
        else:

            self.worker_types[rand_envs] = RolloutWorkerModes.ADR_ROLLOUT
        
        # ensure tensors get re-sampled before new episode
        for k in self.adr_tensor_values:
            self.sample_adr_tensor(k, rand_envs)


    def apply_randomizations(self, dr_params, randomize_buf, adr_objective=None, randomisation_callback=None):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
            randomize_buf: selective randomisation of environments
            adr_objective: consecutive successes scalar
            randomisation_callback: callbacks we may want to use from the environment class
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)

        # for ADR 
        if self.use_adr:
            
            if self.first_randomization:
                adr_env_ids  = list(range(self.num_envs))
            else:
                adr_env_ids = torch.nonzero(randomize_buf, as_tuple=False).squeeze(-1).tolist()
            self.adr_update(adr_env_ids, adr_objective)
            current_adr_params = self.get_current_adr_params(dr_params)

            if self.first_randomization:
                do_nonenv_randomize = True
                env_ids = list(range(self.num_envs))
            else:
                do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
                                
                env_ids = torch.nonzero(randomize_buf, as_tuple=False).squeeze(-1).tolist()
            if do_nonenv_randomize:
                self.last_rand_step = self.last_step            

        # For Manual DR 
        if not self.use_adr:

            if self.first_randomization:
                do_nonenv_randomize = True
                env_ids = list(range(self.num_envs))
            else:
                # randomise if the number of steps since the last randomization is greater than the randomization frequency
                do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
                rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
                rand_envs = torch.logical_and(rand_envs, self.reset_buf)
                env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
                               
                self.randomize_buf[rand_envs] = 0

            if do_nonenv_randomize:
                self.last_rand_step = self.last_step
            
        # We don't use it for ADR(!)
        if self.randomize_act_builtin:
            self.action_randomizations = self.get_randomization_dict(dr_params['actions'], (self.num_actions,))
        
        if self.use_dict_obs and self.randomize_obs_builtin: 
            for nonphysical_param in self.randomisation_obs:
                self.obs_randomizations[nonphysical_param] = self.get_randomization_dict(dr_params['observations'][nonphysical_param], 
                                                                self.obs_space[nonphysical_param].shape)
        elif self.randomize_obs_builtin:
            self.observation_randomizations = self.get_randomization_dict(dr_params['observations'], self.obs_space.shape)


        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)
        

        # Randomize non-environment parameters e.g. gravity, timestep, rest_offset etc.
        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            # Get the list of original paramters set in the yaml and we do add/scale 
            # on these values
            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            # Get prop attrs randomised by add/scale of the original_props values
            # attr is [gravity, reset_offset, ... ]
            # attr_randomization_params can be {'range': [0, 0.5], 'operation': 'additive', 'distribution': 'gaussian'}
            # therefore, prop.val = original_val <operator> random sample 
            # where operator is add/mul 
            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)
                if attr == "gravity":
                    randomisation_callback('gravity', prop.gravity)

            # Randomize physical environments
            # if self.last_step % 10 == 0 and self.last_step > 0:
            #     print('random rest offset = ', prop.physx.rest_offset)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0
        
        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over envs, then loop over actors, then loop over their props 
        # and lastly loop over the ranges of the params 
        for i_, env_id in enumerate(env_ids):

            if self.use_adr:
                # need to generate a custom dictionary for ADR parameters
                env_dr_params = self.get_dr_params_by_env_id(env_id, dr_params, current_adr_params)
            else:
                env_dr_params = dr_params

            for actor, actor_properties in env_dr_params["actor_params"].items():
                if self.first_randomization and i_ % 1000 == 0:
                    print(f'Initializing domain randomization for {actor} env={i_}')


                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]
                
                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}

                for prop_name, prop_attrs in actor_properties.items():

                    # These properties are to do with whole obj mesh related

                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                            
                            if hasattr(self, 'cube_random_params') and actor == 'object':
                                randomisation_callback('scale', new_scale, actor=actor, env_id=env_id)
                            
                            if hasattr(self, 'hand_random_params') and actor == 'object':
                                self.hand_random_params[env_id, 0] = new_scale.mean()
                        continue

                    # Get the properties from the sim API 
                    # prop_names is dof_properties, rigid_body_properties, rigid_shape_properties

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    # if list it is likely to be 
                    #  - rigid_body_properties
                    #  - rigid_shape_properties 

                    if isinstance(prop, list):
                        
                        # Read the original values; remember that 
                        # randomised_prop_val = original_prop_val <operator> random sample

                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]

                        # # list to record value of attr for each body.
                        # recorded_attrs = {"mass": [], "friction": []}


                        # Loop over all the rigid bodies of the actor and then the corresponding 
                        # attribute ranges 
                        for attr, attr_randomization_params_cfg in prop_attrs.items():

                            # for curr_prop, og_p in zip(prop, self.original_props[prop_name]):                            
                            for body_idx, (p, og_p) in enumerate(zip(prop, self.original_props[prop_name])):
                                curr_prop = p 
                            
                                if self.use_adr and isinstance(attr_randomization_params_cfg['range'], dict):
                                    # we have custom ranges for different bodies in this actor
                                    # first: let's find out which group of bodies this body belongs to
                                    body_group_name = None
                                    for group_name, list_of_bodies in self.custom_body_handles[actor].items():
                                        if body_idx in list_of_bodies:
                                            body_group_name = group_name
                                            break
                                    if body_group_name is None:
                                        raise ValueError(
                                            f'Could not find body group for body {body_idx} in actor {actor}.\n'
                                            f'Body groups: {self.custom_body_handles}',
                                        )

                                    # now: get the range for this body group
                                    rand_range = attr_randomization_params_cfg['range'][body_group_name]
                                    attr_randomization_params = copy.deepcopy(attr_randomization_params_cfg)
                                    attr_randomization_params['range'] = rand_range

                                    # we need to sore original params as ADR generated samples need to be bucketed
                                    original_randomization_params = copy.deepcopy(dr_params['actor_params'][actor][prop_name][attr])
                                    original_randomization_params['range'] = original_randomization_params['range'][body_group_name]

                                else:
                                    attr_randomization_params = attr_randomization_params_cfg
                                    # we need to sore original params as ADR generated samples need to be bucketed
                                    original_randomization_params = dr_params['actor_params'][actor][prop_name][attr]

                                assert isinstance(attr_randomization_params['range'], (list, tuple, ListConfig)), \
                                    f'range for {prop_name} must be a list or tuple, got {attr_randomization_params["range"]}'
                            # attrs:
                            #   if rigid_body_properties, it is mass 
                            #   if rigid_shape_properties it is friction etc.

                                setup_only = attr_randomization_params.get('setup_only', False)

                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None

                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], curr_prop, attr)

                                    # generate the samples and add them to props 
                                    # e.g. curr_prop is rigid_body_properties
                                    #      attr is 'mass' (string)
                                    #      mass_val = getattr(curr_prop, 'mass')
                                    #      new_mass_val = mass_val <operator> sample
                                    #      setattr(curr_prop, 'mass', new_mass_val)
                                    apply_random_samples(
                                        curr_prop, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl,
                                        bucketing_randomization_params=original_randomization_params)
                                    
                                    # if attr in recorded_attrs:
                                    #     recorded_attrs[attr] = getattr(curr_prop, attr)

                                    if hasattr(self, 'cube_random_params') and actor == 'object':
                                        assert len(self.original_props[prop_name]) == 1
                                        if attr == 'mass':
                                            self.cube_random_params[env_id, 1] = p.mass
                                        elif attr == 'friction':
                                            self.cube_random_params[env_id, 2] = p.friction
                                else:
                                    set_random_properties = False
                        
                        # # call the callback with the list of attr values that have just been set (for each rigid body / shape in the actor)
                        # for attr, val_list in recorded_attrs.items():
                        #     randomisation_callback(attr, val_list, actor=actor, env_id=env_id)
                    
                    # if it is not a list, it is likely an array 
                    # which means it is for dof_properties 
                    else:
                        
                        # prop_name is e.g. dof_properties with corresponding meta-data 
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        
                        # attrs is damping, stiffness etc.
                        # attrs_randomisation_params is range, distr, schedule 
                        for attr, attr_randomization_params in prop_attrs.items():
                        
                            setup_only = attr_randomization_params.get('setup_only', False)
                        
                            if (setup_only and not self.sim_initialized) or not setup_only:

                                smpl = None
                                
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                
                                # we need to sore original params as ADR generated samples need to be bucketed
                                original_randomization_params = dr_params['actor_params'][actor][prop_name][attr]

                                # generate random samples and add them to props
                                # and we set the props back in sim later on
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl,
                                    bucketing_randomization_params=original_randomization_params)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False
