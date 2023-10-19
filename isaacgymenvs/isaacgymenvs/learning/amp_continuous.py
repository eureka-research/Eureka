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

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 

from tensorboardX import SummaryWriter


class AMPAgent(common_agent.CommonAgent):

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        return

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()
        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        return

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._amp_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            
            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty,
            'disc_logit_loss': disc_logit_loss,
            'disc_agent_acc': disc_agent_acc,
            'disc_demo_acc': disc_demo_acc,
            'disc_agent_logit': disc_agent_logit,
            'disc_demo_logit': disc_demo_logit
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape,
                                                                    device=self.ppo_device)
        
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale
        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return