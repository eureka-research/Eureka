import gym
import numpy as np
from rl_games.common.wrappers import MaskVelocityWrapper

class TestAsymmetricCritic(gym.Env):
    def __init__(self, wrapped_env_name,  **kwargs):
        gym.Env.__init__(self)
        self.apply_mask = kwargs.pop('apply_mask', True)
        self.use_central_value = kwargs.pop('use_central_value', True)
        self.env = gym.make(wrapped_env_name)
        
        if self.apply_mask:
            if wrapped_env_name not in ["CartPole-v1", "Pendulum-v0", "LunarLander-v2", "LunarLanderContinuous-v2"]:
                raise 'unsupported env'    
            self.mask = MaskVelocityWrapper(self.env, wrapped_env_name).mask
        else:
            self.mask = 1

        self.n_agents = 1
        self.use_central_value = True
        self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space
        self.state_space = self.env.observation_space

    def get_number_of_agents(self):
        return self.n_agents

    def reset(self):
        obs = self.env.reset()
        obs_dict = {}
        obs_dict["obs"] = obs * self.mask
        obs_dict["state"] = obs
        if self.use_central_value:
            obses = obs_dict
        else:
            obses = obs_dict["obs"].astype(np.float32)
        return obses

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        obs_dict = {}
        obs_dict["obs"] = obs * self.mask
        obs_dict["state"] = obs
        if self.use_central_value:
            obses = obs_dict
        else:
            obses = obs_dict["obs"].astype(np.float32)
        return obses, rewards, dones, info
    
    def has_action_mask(self):
        return False
