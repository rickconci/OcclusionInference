import torch
import torch.nn as nn
import torch.nn.init as init

class Lin_model(nn.Module):
    def __init__(self,z_in, z_out):
        super(Lin_model, self).__init__()
        self.readout = nn.Linear(z_in, z_out)
        nn.init.kaiming_uniform_(self.readout.weight)

    def forward(self, z_in ):
        z_out = self.readout(z_in)
        return z_out