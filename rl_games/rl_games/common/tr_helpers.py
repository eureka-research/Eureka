import numpy as np
from collections import defaultdict

class LinearValueProcessor:
    def __init__(self, start_eps, end_eps, end_eps_frames):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.end_eps_frames = end_eps_frames
    
    def __call__(self, frame):
        if frame >= self.end_eps_frames:
            return self.end_eps
        df = frame / self.end_eps_frames
        return df * self.end_eps + (1.0 - df) * self.start_eps

class DefaultRewardsShaper:
    def __init__(self, scale_value = 1, shift_value = 0, min_val=-np.inf, max_val=np.inf, log_val=False, is_torch=True):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.log_val = log_val
        self.is_torch = is_torch
        
        if self.is_torch:
            import torch
            self.log = torch.log
            self.clip = torch.clamp
        else:
            self.log = np.log
            self.clip = np.clip

    def __call__(self, reward):
        orig_reward = reward
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        reward = self.clip(reward, self.min_val, self.max_val)

        if self.log_val:
            reward = self.log(reward)
        return reward


def dicts_to_dict_with_arrays(dicts, add_batch_dim = True):
    def stack(v):
        if len(np.shape(v)) == 1:
            return np.array(v)
        else: 
            return np.stack(v)

    def concatenate(v):
        if len(np.shape(v)) == 1:
            return np.array(v)
        else: 
            return np.concatenate(v)

    dicts_len = len(dicts)
    if(dicts_len <= 1):
        return dicts
    res = defaultdict(list)
    { res[key].append(sub[key]) for sub in dicts for key in sub }
    if add_batch_dim:
        concat_func = stack
    else:
        concat_func = concatenate

    res = {k : concat_func(v)  for k,v in res.items()}
    return res

def unsqueeze_obs(obs):
    if type(obs) is dict:
        for k,v in obs.items():
            obs[k] = unsqueeze_obs(v)
    else:
        if len(obs.size()) > 1 or obs.size()[0] > 1:
            obs = obs.unsqueeze(0)
    return obs

def flatten_first_two_dims(arr):
    if arr.ndim > 2:
        return arr.reshape(-1, *arr.shape[-(arr.ndim-2):])
    else:
        return arr.reshape(-1)

def free_mem():
    import ctypes
    ctypes.CDLL('libc.so.6').malloc_trim(0) 