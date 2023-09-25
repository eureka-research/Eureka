from rl_games.common.ivecenv import IVecEnv
import gym
import torch
import numpy as np


class CuleEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        import torchcule
        from torchcule.atari import Env as AtariEnv

        self.batch_size = num_actors
        env_name=kwargs.pop('env_name')
        self.has_lives = kwargs.pop('has_lives', False)
        self.device = kwargs.pop('device', 'cuda:0')
        self.episodic_life = kwargs.pop('episodic_life', False)
        self.use_dict_obs_space = kwargs.pop('use_dict_obs_space', False)
        self.env = AtariEnv(env_name, num_actors, color_mode='gray', repeat_prob=0.0, device=self.device, rescale=True, episodic_life=self.episodic_life, frameskip=4)
        if self.use_dict_obs_space:
            self.observation_space= gym.spaces.Dict({
                'observation' : self.env.observation_space,
                'reward' : gym.spaces.Box(low=0, high=1, shape=( ), dtype=np.float32),
                'last_action': gym.spaces.Box(low=0, high=self.env.action_space.n, shape=(), dtype=int)
            })
        else:
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8) #self.env.observation_space
        self.ids = np.arange(0, num_actors)
        self.action_space = self.env.action_space
        self.scores = np.zeros(num_actors)
        self.returned_scores = np.zeros(num_actors)

    def _set_scores(self, infos, dones):
        # thanks to cleanrl: https://github.com/vwxyzjn/cleanrl/blob/3d20d11f45a5f1d764934e9851b816d0b03d2d10/cleanrl/ppo_atari_envpool.py#L111
        if 'reward' not in infos:
            return
        self.scores += infos["reward"]
        self.returned_scores[:] = self.scores
        infos["scores"] = self.returned_scores

        if self.has_lives:
            all_lives_exhausted = infos["lives"] == 0
            self.scores *= 1 - all_lives_exhausted
        else:
            # removing lives otherwise default observer will use them
            if 'lives' in infos:
                del infos['lives']
            self.scores *= 1 - dones

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        #print(next_obs.size(), 'step!')
        #info['time_outs'] = info['TimeLimit.truncated']
        #self._set_scores(info, is_done)
        if self.use_dict_obs_space:
            next_obs = {
                'observation': next_obs,
                'reward': torch.clip(reward, -1, 1),
                'last_action': action
            }
        return next_obs, reward, is_done, info

    def reset(self):
        obs = self.env.reset()
        #print(obs.size(), 'reset!')
        if self.use_dict_obs_space:
            obs = {
                'observation': obs,
                'reward': torch.zeros(obs.shape[0], device=self.device),
                'last_action': torch.zeros(obs.shape[0], device=self.device),
            }
        return obs

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_cule(**kwargs):
    return CuleEnv("", kwargs.pop('num_actors', 16), **kwargs)