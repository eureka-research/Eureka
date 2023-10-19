from rl_games.common.ivecenv import IVecEnv
import gym
import numpy as np
import torch.utils.dlpack as tpack

def jax_to_torch(tensor):
    from jax._src.dlpack import (to_dlpack,)
    tensor = to_dlpack(tensor)
    tensor = tpack.from_dlpack(tensor)
    return tensor

def torch_to_jax(tensor):
    from jax._src.dlpack import (from_dlpack,)
    tensor = tpack.to_dlpack(tensor)
    tensor = from_dlpack(tensor)
    return tensor


class BraxEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        from brax import envs
        import jax.numpy as jnp

        self.batch_size = num_actors
        env_name=kwargs.pop('env_name', 'ant')
        self.env = envs.create_gym_env(env_name=env_name,
                   batch_size= self.batch_size,
                   seed = 0,
                   backend = 'gpu'
                   )

        obs_high = np.inf * np.ones(self.env._env.unwrapped.observation_size)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        action_high = np.ones(self.env._env.unwrapped.action_size)
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def step(self, action):
        action = torch_to_jax(action)
        next_obs, reward, is_done, info = self.env.step(action)
        next_obs = jax_to_torch(next_obs)
        reward = jax_to_torch(reward)
        is_done = jax_to_torch(is_done)
        return next_obs, reward, is_done, info

    def reset(self):
        # todo add random init like in collab examples?
        obs = self.env.reset()
        return jax_to_torch(obs)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info


def create_brax_env(**kwargs):
    return BraxEnv("", kwargs.pop('num_actors', 256), **kwargs)