from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import  SquashedNormal
from rl_games.common.layers.recurrent import  GRUWithDones, LSTMWithDones
from rl_games.common.layers.value import  TwoHotEncodedValue, DefaultValue
from rl_games.algos_torch.layers import symexp, symlog

def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)

class NetworkBuilder:
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        pass

    def build(self, name, **kwargs):
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

            self.init_factory = object_factory.ObjectFactory()
            #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
            self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
            self.init_factory.register_builder('orthogonal', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('default', lambda **kwargs : nn.Identity() )

        def is_separate_critic(self):
            return False

        def get_value_layer(self):
            return self.value

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _calc_input_size(self, input_shape,cnn_layers=None):
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_rnn(self, name, input, units, layers):
            if name == 'identity':
                return torch_ext.IdentityRNN(input, units)
            if name == 'lstm':
                return LSTMWithDones(input_size=input, hidden_size=units, num_layers=layers)
            if name == 'gru':
                return GRUWithDones(input_size=input, hidden_size=units, num_layers=layers)

        def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_only_first_layer=False,
        norm_func_name = None,
        d2rl=False):
            if d2rl:
                act_layers = [self.activations_factory.create(activation) for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_sequential_mlp(input_size, units, activation, dense_func, norm_func_name = None,)

        def _build_conv(self, ctype, **kwargs):
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn2d(**kwargs)
            if ctype == 'coord_conv2d':
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None):
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_value_layer(self, input_size, output_size, value_type='legacy'):
            if value_type == 'legacy':
                return torch.nn.Linear(input_size, output_size)
            if value_type == 'default':
                return DefaultValue(input_size, output_size)            
            if value_type == 'twohot_encoded':
                return TwoHotEncodedValue(input_size, output_size)

            raise ValueError('value type is not "default", "legacy" or "two_hot_encoded"')



class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            if self.has_cnn:
                if self.permute_input:
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv( **cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)
                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)                

                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.conv = Conv2dAuto(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, bias=not use_bn)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='relu', use_bn=False, use_zero_init=False, use_attention=False):
        super().__init__()
        self.use_zero_init=use_zero_init
        self.use_attention = use_attention
        if use_zero_init:
            self.alpha = nn.Parameter(torch.zeros(1))
        self.activation = activation
        self.conv1 = ConvBlock(channels, channels, use_bn)
        self.conv2 = ConvBlock(channels, channels, use_bn)
        self.activate1 = nn.ReLU()
        self.activate2 = nn.ReLU()
        if use_attention:
            self.ca = ChannelAttention(channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.activate1(x)
        x = self.conv1(x)
        x = self.activate2(x)
        x = self.conv2(x)
        if self.use_attention:
            x = self.ca(x) * x
            x = self.sa(x) * x
        if self.use_zero_init:
            x = x * self.alpha + residual
        else:
            x = x + residual
        return x


class ImpalaSequential(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', use_bn=False, use_zero_init=False):
        super().__init__()    
        self.conv = ConvBlock(in_channels, out_channels, use_bn)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels, activation=activation, use_bn=use_bn, use_zero_init=use_zero_init)
        self.res_block2 = ResidualBlock(out_channels, activation=activation, use_bn=use_bn, use_zero_init=use_zero_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

class A2CResnetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.actions_num = actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            if type(input_shape) is dict:
                input_shape = input_shape['observation']
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            if self.permute_input:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)

            self.cnn = self._build_impala(input_shape, self.conv_depths)
            mlp_input_shape = self._calc_input_size(input_shape, self.cnn)

            in_mlp_shape = mlp_input_shape

            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size =  out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size =  in_mlp_shape
                    in_mlp_shape = self.rnn_units
                if self.require_rewards:
                    rnn_in_size += 1
                if self.require_last_actions:
                    rnn_in_size += actions_num
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                #self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' :self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.flatten_act = self.activations_factory.create(self.activation) 
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            for m in self.mlp:
                if isinstance(m, nn.Linear):    
                    mlp_init(m.weight)

            if self.is_discrete:
                mlp_init(self.logits.weight)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            mlp_init(self.value.weight)     

        def forward(self, obs_dict):
            if self.require_rewards or self.require_last_actions:
                obs = obs_dict['obs']['observation']
                reward = obs_dict['obs']['reward']
                last_action = obs_dict['obs']['last_action']
                if self.is_discrete:
                    last_action = torch.nn.functional.one_hot(last_action.long(), num_classes=self.actions_num)
            else:
                obs = obs_dict['obs']
            if self.permute_input:
                obs = obs.permute((0, 3, 1, 2))

            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            out = obs
            out = self.cnn(out)
            out = out.flatten(1)         
            out = self.flatten_act(out)

            if self.has_rnn:
                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.mlp(out)

                obs_list = [out]
                if self.require_rewards:
                    obs_list.append(reward.unsqueeze(1))
                if self.require_last_actions:
                    obs_list.append(last_action)
                out = torch.cat(obs_list, dim=1)
                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.mlp(out)

            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states

        def load(self, params):
            self.separate = False
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            self.is_multi_discrete = 'multi_discrete'in params['space']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']    
            self.has_rnn = 'rnn' in params
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_ln = params['rnn'].get('layer_norm', False)
            self.has_cnn = True
            self.permute_input = params['cnn'].get('permute_input', True)
            self.conv_depths = params['cnn']['conv_depths']
            self.require_rewards = params.get('require_rewards')
            self.require_last_actions = params.get('require_last_actions')

        def _build_impala(self, input_shape, depths):
            in_channels = input_shape[0]
            layers = nn.ModuleList()    
            for d in depths:
                layers.append(ImpalaSequential(in_channels, d))
                in_channels = d
            return nn.Sequential(*layers)

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)), 
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)))                

    def build(self, name, **kwargs):
        net = A2CResnetBuilder.Network(self.params, **kwargs)
        return net


class DiagGaussianActor(NetworkBuilder.BaseNetwork):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, output_dim, log_std_bounds, **mlp_args):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.trunk = self._build_mlp(**mlp_args)
        last_layer = list(self.trunk.children())[-2].out_features
        self.trunk = nn.Sequential(*list(self.trunk.children()), nn.Linear(last_layer, output_dim))

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        #log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        #log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # TODO: Refactor

        dist = SquashedNormal(mu, std)
        # Modify to only return mu and std
        return dist


class DoubleQCritic(NetworkBuilder.BaseNetwork):
    """Critic network, employes double Q-learning."""
    def __init__(self, output_dim, **mlp_args):
        super().__init__()

        self.Q1 = self._build_mlp(**mlp_args)
        last_layer = list(self.Q1.children())[-2].out_features
        self.Q1 = nn.Sequential(*list(self.Q1.children()), nn.Linear(last_layer, output_dim))

        self.Q2 = self._build_mlp(**mlp_args)
        last_layer = list(self.Q2.children())[-2].out_features
        self.Q2 = nn.Sequential(*list(self.Q2.children()), nn.Linear(last_layer, output_dim))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class SACBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = SACBuilder.Network(self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            mlp_input_shape = input_shape

            actor_mlp_args = {
                'input_size' : obs_dim, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }

            critic_mlp_args = {
                'input_size' : obs_dim + action_dim, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            print("Building Actor")
            self.actor = self._build_actor(2*action_dim, self.log_std_bounds, **actor_mlp_args)

            if self.separate:
                print("Building Critic")
                self.critic = self._build_critic(1, **critic_mlp_args)
                print("Building Critic Target")
                self.critic_target = self._build_critic(1, **critic_mlp_args)
                self.critic_target.load_state_dict(self.critic.state_dict())  

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_critic(self, output_dim, **mlp_args):
            return DoubleQCritic(output_dim, **mlp_args)

        def _build_actor(self, output_dim, log_std_bounds, **mlp_args):
            return DiagGaussianActor(output_dim, log_std_bounds, **mlp_args)

        def forward(self, obs_dict):
            """TODO"""
            obs = obs_dict['obs']
            mu, sigma = self.actor(obs)
            return mu, sigma
 
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.separate = params.get('separate', True)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.log_std_bounds = params.get('log_std_bounds', None)

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False

