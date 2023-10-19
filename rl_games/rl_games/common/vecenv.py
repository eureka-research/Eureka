import ray
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.env_configurations import configurations
from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
import numpy as np
import gym
import random
from time import sleep
import torch

class RayWorker:
    def __init__(self, config_name, config):
        self.env = configurations[config_name]['env_creator'](**config)

    def _obs_to_fp32(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        if dv.dtype == np.float64:
                            v[dk] = dv.astype(np.float32)
                else:
                    if v.dtype == np.float64:
                        obs[k] = v.astype(np.float32)
        else:
            if obs.dtype == np.float64:
                obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        
        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        return next_state, reward, is_done, info

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
            
    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_to_fp32(obs)
        return obs

    def get_action_mask(self):
        return self.env.get_action_mask()

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        else:
            return 1

    def set_weights(self, weights):
        self.env.update_weights(weights)

    def can_concat_infos(self):
        if hasattr(self.env, 'concat_infos'):
            return self.env.concat_infos
        else:
            return False

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        #if isinstance(observation_space, gym.spaces.dict.Dict):
        #    observation_space = observation_space['observations']

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if hasattr(self.env, 'use_central_value'):
            info['use_global_observations'] = self.env.use_central_value
        if hasattr(self.env, 'value_size'):
            info['value_size'] = self.env.value_size
        if hasattr(self.env, 'state_space'):
            info['state_space'] = self.env.state_space
        return info


class RayVecEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)
        self.remote_worker = ray.remote(RayWorker)
        self.workers = [self.remote_worker.remote(self.config_name, kwargs) for i in range(self.num_actors)]

        if self.seed is not None:
            seeds = range(self.seed, self.seed + self.num_actors)
            seed_set = []
            for (seed, worker) in zip(seeds, self.workers):	        
                seed_set.append(worker.seed.remote(seed))
            ray.get(seed_set)

        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = ray.get(res)

        res = self.workers[0].get_env_info.remote()
        env_info = ray.get(res)
        res = self.workers[0].can_concat_infos.remote()
        can_concat_infos = ray.get(res)
        self.use_global_obs = env_info['use_global_observations']
        self.concat_infos = can_concat_infos
        self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
        self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
        if self.num_agents == 1:
            self.concat_func = np.stack
        else:
            self.concat_func = np.concatenate
    
    def step(self, actions):
        newobs, newstates, newrewards, newdones, newinfos = [], [], [], [], []
        res_obs = []
        if self.num_agents == 1:
            for (action, worker) in zip(actions, self.workers):	        
                res_obs.append(worker.step.remote(action))
        else:
            for num, worker in enumerate(self.workers):
                res_obs.append(worker.step.remote(actions[self.num_agents * num: self.num_agents * num + self.num_agents]))

        all_res = ray.get(res_obs)
        for res in all_res:
            cobs, crewards, cdones, cinfos = res
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        if self.concat_infos:
            newinfos = dicts_to_dict_with_arrays(newinfos, False)
        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

    def get_env_info(self):
        res = self.workers[0].get_env_info.remote()
        return ray.get(res)

    def set_weights(self, indices, weights):
        res = []
        for ind in indices:
            res.append(self.workers[ind].set_weights.remote(weights))
        ray.get(res)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self):
        res_obs = [worker.reset.remote() for worker in self.workers]
        newobs, newstates = [],[]
        for res in res_obs:
            cobs = ray.get(res)
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        return ret_obs

vecenv_config = {}

def register(config_name, func):
    vecenv_config[config_name] = func

def create_vec_env(config_name, num_actors, **kwargs):
    vec_env_name = configurations[config_name]['vecenv_type']
    return vecenv_config[vec_env_name](config_name, num_actors, **kwargs)

register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))

from rl_games.envs.brax import BraxEnv
register('BRAX', lambda config_name, num_actors, **kwargs: BraxEnv(config_name, num_actors, **kwargs))

from rl_games.envs.envpool import Envpool
register('ENVPOOL', lambda config_name, num_actors, **kwargs: Envpool(config_name, num_actors, **kwargs))

from rl_games.envs.cule import CuleEnv
register('CULE', lambda config_name, num_actors, **kwargs: CuleEnv(config_name, num_actors, **kwargs))