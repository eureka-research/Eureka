import gym
import numpy as np


class ExampleEnv(gym.Env):
    '''
    Just example empty env which demonstrates additional features compared to the default openai gym
    '''
    def __init__(self,  **kwargs):
        gym.Env.__init__(self)

        self.use_central_value = True
        self.value_size = 2
        self.concat_infos = False
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2),gym.spaces.Discrete(3)]) # gym.spaces.Discrete(3), gym.spaces.Box(low=0, high=1, shape=(3, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32) # or Dict

    def get_number_of_agents(self):
        return 1

    def has_action_mask(self):
        return False

    def get_action_mask(self):
        pass