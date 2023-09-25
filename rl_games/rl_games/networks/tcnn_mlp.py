import torch
from torch import nn
import torch.nn.functional as F


class TcnnNet(nn.Module):
    def __init__(self, params, **kwargs):
        import tinycudann as tcnn
        nn.Module.__init__(self)
        self.actions_num = actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = input_shape[0]
        self.central_value = params.get('central_value', False)
        self.sigma = torch.nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                        requires_grad=True)
        self.model = tcnn.NetworkWithInputEncoding(n_input_dims=num_inputs, n_output_dims=actions_num+1,
                                              encoding_config=params["encoding"], network_config=params["network"])
    def is_rnn(self):
        return False

    def forward(self, obs):
        obs = obs['obs']
        mu_val = self.model(obs)
        mu, value = torch.split(mu_val, [self.actions_num, 1], dim=1)
        return mu, mu * 0.0 + self.sigma, value, None


from rl_games.algos_torch.network_builder import NetworkBuilder


class TcnnNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TcnnNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)
