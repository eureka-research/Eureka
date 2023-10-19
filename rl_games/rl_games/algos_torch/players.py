from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()

class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic = True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = False
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
