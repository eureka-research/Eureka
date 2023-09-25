import gym
import numpy as np
from pettingzoo.classic import connect_four_v0
import yaml
from rl_games.torch_runner import Runner
import os
from collections import deque


class ConnectFourSelfPlay(gym.Env):
    def __init__(self, name="connect_four_v0",  **kwargs):
        gym.Env.__init__(self)
        self.name = name
        self.is_deterministic = kwargs.pop('is_deterministic', False)
        self.is_human = kwargs.pop('is_human', False)
        self.random_agent = kwargs.pop('random_agent', False)
        self.config_path = kwargs.pop('config_path')
        self.agent = None

        self.env = connect_four_v0.env()  # gym.make(name, **kwargs)
        self.action_space = self.env.action_spaces['player_0']
        observation_space = self.env.observation_spaces['player_0']
        shp = observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(shp[:-1] + (shp[-1] * 2,)), dtype=np.uint8)
        self.obs_deque = deque([], maxlen=2)
        self.agent_id = 0

    def _get_legal_moves(self, agent_id):
        name = 'player_0' if agent_id == 0 else 'player_1'
        action_ids = self.env.infos[name]['legal_moves']
        mask = np.zeros(self.action_space.n, dtype=np.bool)
        mask[action_ids] = True
        return mask, action_ids

    def env_step(self, action):
        obs = self.env.step(action)
        info = {}
        name = 'player_0' if self.agent_id == 0 else 'player_1'
        reward = self.env.rewards[name]
        done = self.env.dones[name]
        return obs, reward, done, info

    def get_obs(self):
        return np.concatenate(self.obs_deque, -1).astype(np.uint8) * 255

    def reset(self):
        if self.agent == None:
            self.create_agent(self.config_path)

        self.agent_id = np.random.randint(2)
        obs = self.env.reset()
        self.obs_deque.append(obs)
        self.obs_deque.append(obs)
        if self.agent_id == 1:
            op_obs = self.get_obs()
            op_obs = self.agent.obs_to_torch(op_obs)
            mask, ids = self._get_legal_moves(0)
            if self.is_human:
                self.render()
                opponent_action = int(input())
            else:
                if self.random_agent:
                    opponent_action = np.random.choice(ids, 1)[0]
                else:
                    opponent_action = self.agent.get_masked_action(
                        op_obs, mask, self.is_deterministic).item()

            obs, _, _, _ = self.env_step(opponent_action)

            self.obs_deque.append(obs)
        return self.get_obs()

    def create_agent(self, config):
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)
            runner = Runner()
            runner.load(config)
        config = runner.get_prebuilt_config()
        # 'RAYLIB has bug here, CUDA_VISIBLE_DEVICES become unset'
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            os.environ.pop('CUDA_VISIBLE_DEVICES')

        self.agent = runner.create_player()
        self.agent.model.eval()

    def step(self, action):

        obs, reward, done, info = self.env_step(action)
        self.obs_deque.append(obs)

        if done:
            if reward == 1:
                info['battle_won'] = 1
            else:
                info['battle_won'] = 0
            return self.get_obs(), reward, done, info

        op_obs = self.get_obs()

        op_obs = self.agent.obs_to_torch(op_obs)
        mask, ids = self._get_legal_moves(1-self.agent_id)
        if self.is_human:
            self.render()
            opponent_action = int(input())
        else:
            if self.random_agent:
                opponent_action = np.random.choice(ids, 1)[0]
            else:
                opponent_action = self.agent.get_masked_action(
                    op_obs, mask, self.is_deterministic).item()
        obs, reward, done, _ = self.env_step(opponent_action)
        if done:
            if reward == -1:
                info['battle_won'] = 0
            else:
                info['battle_won'] = 1
        self.obs_deque.append(obs)
        return self.get_obs(), reward, done, info

    def render(self, mode='ansi'):
        self.env.render(mode)

    def update_weights(self, weigths):
        self.agent.set_weights(weigths)

    def get_action_mask(self):
        mask, _ = self._get_legal_moves(self.agent_id)
        return mask

    def has_action_mask(self):
        return True
