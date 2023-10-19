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
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

import isaacgymenvs.learning.common_agent as common_agent 
import isaacgymenvs.learning.gen_amp as gen_amp
import isaacgymenvs.learning.gen_amp_models as gen_amp_models
import isaacgymenvs.learning.gen_amp_network_builder as gen_amp_network_builder

from tensorboardX import SummaryWriter


class HRLAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        return

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        rewards = 0.0
        done_count = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, actions)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

        rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = gen_amp_network_builder.GenAMPBuilder()
        network_builder.load(network_params)

        network = gen_amp_models.ModelGenAMPContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = gen_amp.GenAMPAgent('llc', llc_agent_config)
        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer' : self.algo_observer}
        config['env_info'] = llc_env_info

        return config

    def _compute_llc_action(self, obs, actions):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)
        z = torch.nn.functional.normalize(actions, dim=-1)

        mu, _ = self._llc_agent.model.a2c_network.eval_actor(obs=processed_obs, amp_latents=z)
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs
