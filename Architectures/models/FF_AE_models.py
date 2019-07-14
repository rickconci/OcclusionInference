
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
    
class conv_AE(nn.Module):
    def __init__(self, z_dim=10,n_filter=32, nc=1,l1weight=0.05, train=True):
        super(conv_AE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.n_filter = n_filter
        self.train = train
        self.l1weight = l1weight
        #assume initial size is 64 x 64 
        self.encoder = nn.Sequential(
            #nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            #nn.ReLU(True),
            nn.Conv2d(nc, self.n_filter, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(self.n_filter, self.n_filter, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(self.n_filter, self.n_filter, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, self.n_filter*4*4)),                  # B, 512
            nn.Linear(self.n_filter*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, self.n_filter*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, self.n_filter, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(self.n_filter, self.n_filter, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n_filter, self.n_filter, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(self.n_filter, nc, 4, 2, 1), # B,  32, 32, 32
            #nn.ReLU(True),
            #nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )
        self.weight_init() 
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, train=True ):
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

    
class conv_AE_mini(nn.Module):
    def __init__(self, z_dim=10,n_filter=10, nc=1,l1weight=0.05, train=True):
        super(conv_AE_mini, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.n_filter = n_filter
        self.train = train
        self.l1weight = l1weight
        #assume initial size is 32 x 32 
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, self.n_filter, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            View((-1, self.n_filter*16*16)),                  # B, 512
            nn.Linear(self.n_filter*16*16, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, self.n_filter*16*16),              # B, 512
            nn.ReLU(True),
            View((-1, self.n_filter, 16, 16)),                # B,  32,  4,  4
            nn.ConvTranspose2d(self.n_filter, nc, 4, 2, 1), # B,  32, 32, 32
        )
        self.weight_init() 
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
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
   