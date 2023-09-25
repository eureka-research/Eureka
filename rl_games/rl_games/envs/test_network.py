import torch
from torch import nn
import torch.nn.functional as F

 
class TestNet(nn.Module):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = 0

        assert(type(input_shape) is dict)
        for k,v in input_shape.items():
            num_inputs +=v[0]

        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.mean_linear = nn.Linear(64, actions_num)
        self.value_linear = nn.Linear(64, 1)

    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        obs = torch.cat([obs['pos'], obs['info']], axis=-1)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        action = self.mean_linear(x)
        value = self.value_linear(x)
        if self.central_value:
            return value, None
        return action, value, None


from rl_games.algos_torch.network_builder import NetworkBuilder

class TestNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TestNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
