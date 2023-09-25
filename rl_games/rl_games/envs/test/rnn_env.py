import gym
import numpy as np


class TestRNNEnv(gym.Env):
    def __init__(self,  **kwargs):
        gym.Env.__init__(self)
 
        self.obs_dict = {}
        self.max_steps = kwargs.pop('max_steps', 21)
        self.show_time = kwargs.pop('show_time', 1)
        self.min_dist = kwargs.pop('min_dist', 2)
        self.max_dist = kwargs.pop('max_dist', 8)
        self.hide_object = kwargs.pop('hide_object', False)
        self.use_central_value = kwargs.pop('use_central_value', False)
        self.apply_dist_reward = kwargs.pop('apply_dist_reward', False)
        self.apply_exploration_reward = kwargs.pop('apply_exploration_reward', False)
        self.multi_head_value = kwargs.pop('multi_head_value', False)
        if self.multi_head_value:
            self.value_size = 2
        else:
            self.value_size = 1

        self.multi_discrete_space = kwargs.pop('multi_discrete_space', False)
        if self.multi_discrete_space:
            self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2),gym.spaces.Discrete(3)])
        else:
            self.action_space = gym.spaces.Discrete(4)

        self.multi_obs_space = kwargs.pop('multi_obs_space', False)
        if self.multi_obs_space:
            spaces = {
                'pos': gym.spaces.Box(low=0, high=1, shape=(2, ), dtype=np.float32),
                'info': gym.spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float32),
            }
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32)

        self.state_space = self.observation_space
        if self.apply_exploration_reward:
            pass
        self.reset()

    def get_number_of_agents(self):
        return 1

    def reset(self):
        self._curr_steps = 0
        self._current_pos = [0,0]
        bound = self.max_dist - self.min_dist
        rand_dir = - 2 * np.random.randint(0, 2, (2,)) + 1
        self._goal_pos = rand_dir * np.random.randint(self.min_dist, self.max_dist+1, (2,))
        obs = np.concatenate([self._current_pos, self._goal_pos, [1, 0]], axis=None)
        obs = obs.astype(np.float32)
        if self.multi_obs_space:
            obs = {
                'pos': obs[:2],
                'info': obs[2:]
            }
        if self.use_central_value:
            obses = {}
            obses["obs"] = obs
            obses["state"] = obs
        else:
            obses = obs
        return obses

    def step_categorical(self, action):
        if self._curr_steps > 1:
            if action == 0:
                self._current_pos[0] += 1
            if action == 1:
                self._current_pos[0] -= 1
            if action == 2:
                self._current_pos[1] += 1
            if action == 3:
                self._current_pos[1] -= 1   

    def step_multi_categorical(self, action):
        if self._curr_steps > 1:
            if action[0] == 0:
                self._current_pos[0] += 1
            if action[0] == 1:
                self._current_pos[0] -= 1
            if action[1] == 0:
                self._current_pos[1] += 1
            if action[1] == 1:
                self._current_pos[1] -= 1
            if action[1] == 2:
                pass

    def step(self, action):
        info = {}  
        self._curr_steps += 1
        if self.multi_discrete_space:
            self.step_multi_categorical(action)
        else:
            self.step_categorical(action)
        reward = [0.0, 0.0]
        done = False
        dist = self._current_pos - self._goal_pos
        if (dist**2).sum() < 0.0001:
            reward[0] = 1.0
            info = {'scores' : 1} 
            done = True
        elif self._curr_steps == self.max_steps:
            info = {'scores' : 0} 
            done = True

        dist_coef = -0.1
        if self.apply_dist_reward:
            reward[1] = dist_coef * np.abs(dist).sum() / self.max_dist

        show_object = 0
        if self.hide_object:
            obs = np.concatenate([self._current_pos, [0,0], [show_object, self._curr_steps]], axis=None)
        else:
            show_object = 1
            obs = np.concatenate([self._current_pos, self._goal_pos, [show_object, self._curr_steps]], axis=None)
        obs = obs.astype(np.float32)
        #state = state.astype(np.float32)
        if self.multi_obs_space:
            obs = {
                'pos': obs[:2],
                'info': obs[2:]
            }
        if self.use_central_value:
            state = np.concatenate([self._current_pos, self._goal_pos, [show_object, self._curr_steps]], axis=None)
            obses = {}
            obses["obs"] = obs
            if self.multi_obs_space:
                obses["state"] = {
                    'pos': state[:2],
                    'info': state[2:]
                }
            else:
                obses["state"] = state.astype(np.float32)
        else:
            obses = obs
        if self.multi_head_value:
            pass
        else:
            reward = reward[0] + reward[1]
        
        return obses, np.array(reward).astype(np.float32), done, info
    
    def has_action_mask(self):
        return False