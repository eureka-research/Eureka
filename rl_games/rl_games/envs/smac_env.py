import gym
import numpy as np
from smac.env import StarCraft2Env
from smac.env import MultiAgentEnv

class SMACEnv(gym.Env):
    def __init__(self, name="3m",  **kwargs):
        gym.Env.__init__(self)
        self._seed = kwargs.pop('seed', None)
        self.reward_sparse = kwargs.get('reward_sparse', False)
        self.use_central_value = kwargs.pop('central_value', False)
        self.concat_infos = True
        self.random_invalid_step = kwargs.pop('random_invalid_step', False)
        self.replay_save_freq = kwargs.pop('replay_save_freq', 10000)
        self.apply_agent_ids = kwargs.pop('apply_agent_ids', True)
        self.env = StarCraft2Env(map_name=name, seed=self._seed, **kwargs)
        self.env_info = self.env.get_env_info()

        self._game_num = 0
        self.n_actions = self.env_info["n_actions"]
        self.n_agents = self.env_info["n_agents"]
        self.action_space = gym.spaces.Discrete(self.n_actions)
        one_hot_agents = 0

        if self.apply_agent_ids:
            one_hot_agents = self.n_agents
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.env_info['obs_shape']+one_hot_agents, ), dtype=np.float32)
        self.state_space = gym.spaces.Box(low=0, high=1, shape=(self.env_info['state_shape'], ), dtype=np.float32)

        self.obs_dict = {}

    def _preproc_state_obs(self, state, obs):
        # todo: remove from self
        if self.apply_agent_ids:
            num_agents = self.n_agents
            obs = np.array(obs)
            all_ids = np.eye(num_agents, dtype=np.float32)
            obs = np.concatenate([obs, all_ids], axis=-1)

        self.obs_dict["obs"] = np.array(obs)
        self.obs_dict["state"] = np.array(state)

        if self.use_central_value:
            return self.obs_dict
        else:
            return self.obs_dict["obs"]

    def get_number_of_agents(self):
        return self.n_agents

    def reset(self):
        if self._game_num % self.replay_save_freq == 1:
            print('saving replay')
            self.env.save_replay()
        self._game_num += 1
        obs, state = self.env.reset() # rename, to think remove
        obs_dict = self._preproc_state_obs(state, obs)

        return obs_dict

    def _preproc_actions(self, actions):
        actions = actions.copy()
        rewards = np.zeros_like(actions)
        mask = self.get_action_mask()
        for ind, action in enumerate(actions, start=0):
            avail_actions = np.nonzero(mask[ind])[0]
            if action not in avail_actions:
                actions[ind] = np.random.choice(avail_actions)
                #rewards[ind] = -0.05
        return actions, rewards

    def step(self, actions):
        fixed_rewards = None

        if self.random_invalid_step:
            actions, fixed_rewards = self._preproc_actions(actions)

        reward, done, info = self.env.step(actions)
        time_out = self.env._episode_steps >= self.env.episode_limit
        info['time_outs'] = [time_out]*self.n_agents

        if done:
            battle_won = info.get('battle_won', False)
            if not battle_won and self.reward_sparse:
                reward = -1.0

        obs = self.env.get_obs()
        state = self.env.get_state()
        obses = self._preproc_state_obs(state, obs)
        rewards = np.repeat (reward, self.n_agents)
        dones = np.repeat (done, self.n_agents)

        if fixed_rewards is not None:
            rewards += fixed_rewards

        return obses, rewards, dones, info

    def get_action_mask(self):
        return np.array(self.env.get_avail_actions(), dtype=np.bool)
    
    def has_action_mask(self):
        return not self.random_invalid_step

    def seed(self, _):
        pass

