import torch

class D2RLNet(torch.nn.Module):
    def __init__(self, input_size, 
        units, 
        activations,
        norm_func_name = None):
        torch.nn.Module.__init__(self)
        self.activations = torch.nn.ModuleList(activations)
        self.linears = torch.nn.ModuleList([])
        self.norm_layers = torch.nn.ModuleList([])
        self.num_layers = len(units)
        last_size = input_size
        for i in range(self.num_layers):
            self.linears.append(torch.nn.Linear(last_size, units[i]))
            last_size = units[i] + input_size
            if norm_func_name == 'layer_norm':
                self.norm_layers.append(torch.nn.LayerNorm(units[i]))
            elif norm_func_name == 'batch_norm':
                self.norm_layers.append(torch.nn.BatchNorm1d(units[i]))
            else:
                self.norm_layers.append(torch.nn.Identity())

    def forward(self, input):
        x = self.linears[0](input)
        x = self.activations[0](x)
        x = self.norm_layers[0](x)
        for i in range(1,self.num_layers):
            x = torch.cat([x,input], dim=1)
            x = self.linears[i](x)
            x = self.norm_layers[i](x)  
            x = self.activations[i](x)               
        return x