import gym
import numpy as np
import slimevolleygym
import yaml
from rl_games.torch_runner import Runner
import os

class SlimeVolleySelfplay(gym.Env):
    def __init__(self, name="SlimeVolleyDiscrete-v0",  **kwargs):
        gym.Env.__init__(self)
        self.name = name
        self.is_deterministic = kwargs.pop('is_deterministic', False)
        self.config_path = kwargs.pop('config_path')
        self.agent = None
        self.pos_scale = 1
        self.neg_scale =  kwargs.pop('neg_scale', 1)
        self.sum_rewards = 0
        self.env = gym.make(name, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        if self.agent == None:
            self.create_agent(self.config_path)
        obs = self.env.reset()
        self.opponent_obs = obs
        self.sum_rewards = 0
        return obs

    def create_agent(self, config='rl_games/configs/ma/ppo_slime_self_play.yaml'):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)
            runner = Runner()
            from rl_games.common.env_configurations import get_env_info
            config['params']['config']['env_info'] = get_env_info(self)
            runner.load(config)
        config = runner.get_prebuilt_config()

        'RAYLIB has bug here, CUDA_VISIBLE_DEVICES become unset'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.agent = runner.create_player()


    def step(self, action):
        op_obs = self.agent.obs_to_torch(self.opponent_obs)
        
        opponent_action = self.agent.get_action(op_obs, self.is_deterministic).item()
        obs, reward, done, info = self.env.step(action, opponent_action)
        self.sum_rewards += reward
        if reward < 0:
            reward = reward * self.neg_scale
        
        self.opponent_obs = info['otherObs']
        if done:
            info['battle_won'] = np.sign(self.sum_rewards)
        return obs, reward, done, info

    def render(self,mode):
        self.env.render(mode)

    def update_weights(self, weigths):
        self.agent.set_weights(weigths)
