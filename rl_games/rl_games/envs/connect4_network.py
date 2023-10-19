import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(4, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s['obs'].contiguous()
        #s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s



class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out



class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(6*7*32, 7)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.relu(self.fc2(v))
        v = torch.tanh(v)
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        return p, v, None

class ConnectNet(nn.Module):
    def __init__(self, blocks):
        super(ConnectNet, self).__init__()
        self.blocks = blocks
        self.conv = ConvBlock()
        for block in range(self.blocks):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    def is_rnn(self):
        return False

    def forward(self,s):
        s = s.permute((0, 3, 1, 2))
        s = self.conv(s)
        for block in range(self.blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


from rl_games.algos_torch.network_builder import NetworkBuilder

class ConnectBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
        self.blocks = params['blocks']

    def build(self, name, **kwargs):
        return ConnectNet(self.blocks)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

