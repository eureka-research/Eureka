from rl_games.networks.tcnn_mlp import TcnnNetBuilder
from rl_games.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)