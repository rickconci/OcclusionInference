
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class L1Penality(torch.autograd.Function):
    """
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    """
    
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input
    
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input, None

class BLT_AE(nn.Module):
    def __init__(self, z_dim=10,n_filter=32, nc=1,l1weight=0.05, train=True):
        super(conv_AE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.n_filter = n_filter
        self.train = train
        self.l1weight = l1weight
        
        self.W_b_1 = nn.Conv2d()
        self.W_l_1 = nn.Conv2d()
        self.W_t_1 = nn.Conv2d()
        self.W_b_2
        self.W_l_2
        self.W_T_2
        self.Lin_1 = nn.Linear()
        
        self.weight_init() 
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, train=True ):
        
        for t in range(4):
            Z_1 += self.W_b_1(x) + self.W_l_1(F.relu(Z_1)) + self.W_T_1(F.relu(Z_2)) 
            Z_2 += self.W_b_2(F.relu(Z1)) + self.W_l_2(F.relu(Z_2)) + self.W_T_2(F.relu(Z_3))
            Z_3 += self.W_b_3(F.relu(Z2)) + self.W_l_3(F.relu(Z_3)) + self.W_T_3(F.relu(Z_4))
        distributions = F.relu(self.Lin_1(Z_3))   
        
        p = distributions[:, :self.z_dim]
        mu = distributions[:, self.z_dim:2*self.z_dim]
        logvar = distributions[:, 2*self.z_dim:]

        z = reparametrize(mu, logvar)
        
        
        
        z = self._encode(x)
        z = L1Penality.apply(z, self.l1weight)
        #print(torch.max(z), torch.min(z), torch.mean(z))
        x_recon = self._decode(z)
        x_recon = x_recon.view(x.size())
        return x_recon
     

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)  

    
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
   