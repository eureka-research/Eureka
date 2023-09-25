import gym
import numpy as np
import os
import random
from diambra_environment.diambraGym import diambraGym
from diambra_environment.makeDiambraEnv import make_diambra_env

class DiambraEnv(gym.Env):
    def __init__(self, **kwargs):
        gym.Env.__init__(self)
        self.seed = kwargs.pop('seed', None)

        self.difficulty = kwargs.pop('difficulty', 3)
        self.env_path = kwargs.pop('env_path', "/home/trrrrr/Documents/github/ml/diambra/DIAMBRAenvironment-main")
        self.character = kwargs.pop('character', 'Raidou')
        self.frame_stack = kwargs.pop('frame_stack', 3)
        self.attacks_buttons = kwargs.pop('attacks_buttons', False)
        self._game_num = 0
        self.n_agents = 1
        self.rank = random.randint(0, 100500)
        repo_base_path = os.path.abspath(self.env_path) # Absolute path to your DIAMBRA environment

        env_kwargs = {}
        env_kwargs["gameId"] = "doapp"
        env_kwargs["roms_path"] = os.path.join(repo_base_path, "roms/") # Absolute path to roms

        env_kwargs["mame_diambra_step_ratio"] = 6
        env_kwargs["render"] = False
        env_kwargs["lock_fps"] = False # Locks to 60 FPS
        env_kwargs["sound"] = env_kwargs["lock_fps"] and env_kwargs["render"]

        env_kwargs["player"] = "Random"

        env_kwargs["difficulty"] = self.difficulty
        env_kwargs["characters"]  = [[self.character, "Random"], [self.character, "Random"]]
        env_kwargs["charOutfits"] = [2, 2]

        gym_kwargs = {}
        gym_kwargs["P2brain"]               = None
        gym_kwargs["continue_game"]         = 0.0
        gym_kwargs["show_final"]            = False
        gym_kwargs["gamePads"]              = [None, None]
        gym_kwargs["actionSpace"]           = ["discrete", "multiDiscrete"]
        #gym_kwargs["attackButCombinations"] = [False, False]
        gym_kwargs["attackButCombinations"] = [self.attacks_buttons, self.attacks_buttons]
        gym_kwargs["actBufLen"]             = 12
        wrapper_kwargs = {}
        wrapper_kwargs["hwc_obs_resize"]    = [128, 128, 1]
        wrapper_kwargs["normalize_rewards"] = True
        wrapper_kwargs["clip_rewards"]      = False
        wrapper_kwargs["frame_stack"]       = self.frame_stack
        wrapper_kwargs["dilation"]          = 1
        wrapper_kwargs["scale"]             = True
        wrapper_kwargs["scale_mod"]         = 0
        key_to_add = []
        key_to_add.append("actionsBuf")
        key_to_add.append("ownHealth")
        key_to_add.append("oppHealth")
        key_to_add.append("ownPosition")
        key_to_add.append("oppPosition")
        key_to_add.append("stage")
        key_to_add.append("character")
        
        self.env = make_diambra_env(diambraGym, env_prefix="Train" + str(self.rank), seed= self.rank,  
            diambra_kwargs=env_kwargs, 
            diambra_gym_kwargs=gym_kwargs,
            wrapper_kwargs=wrapper_kwargs, 
            key_to_add=key_to_add)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


    def _preproc_state_obs(self, obs):
        return obs

    def reset(self):
        self._game_num += 1
        obs = self.env.reset() # rename, to think remove
        obs_dict = self._preproc_state_obs(obs)
        return obs_dict

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)

        return obs, reward, done, info
    
    def has_action_mask(self):
        return False