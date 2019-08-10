import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class NonLin_model(nn.Module):
    def __init__(self,z_in, z_out):
        super(Lin_model, self).__init__()
        self.layer1 = nn.Linear(z_in, 50)
        self.layer2 = nn.Linear(50, z_out)
        nn.init.kaiming_uniform_(self.layer1.weight)
        nn.init.kaiming_uniform_(self.layer2.weight)

    def forward(self, z_in ):
        z_out =self.layer2(F.relu(self.layer1(z_in)))
        return z_out
    
class Lin_model(nn.Module):
    def __init__(self,z_in, z_out):
        super(Lin_model, self).__init__()
        self.layer1 = nn.Linear(z_in, z_out)
        nn.init.kaiming_uniform_(self.layer1.weight)

    def forward(self, z_in ):
        z_out =self.layer1(z_in)
        return z_out