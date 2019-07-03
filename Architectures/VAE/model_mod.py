
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import pickle

def reparametrize_gaussian(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def reparametrize_bernoulli(p_dist):
    eps = Variable(p_dist.data.new(p_dist.size()).uniform_(0,1))
    z = F.sigmoid(torch.log(eps+ 1e-20) - torch.log(1-eps+ 1e-20) + torch.log(p_dist+ 1e-20) - torch.log(1-p_dist+ 1e-20))
    return z


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


    
class gauss_VAE(nn.Module):
    def __init__(self, z_dim=20,n_filter=32, nc=1, train=True):
        super(gauss_VAE, self).__init__()
        self.nc = nc
        self.z_dim_tot = z_dim
        self.z_dim_gauss = z_dim
        self.n_filter = n_filter
        self.train = train
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
            nn.Linear(256, z_dim*2),             # B, z_dim*2
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
        self.train = train
        if self.train==True:
            distributions = self._encode(x)
            mu = distributions[:, :self.z_dim_tot]
            logvar = distributions[:, self.z_dim_tot:]
            z = reparametrize_gaussian(mu, logvar)
            x_recon = self._decode(z)
            x_recon = x_recon.view(x.size())
            return x_recon, mu, logvar
        elif self.train ==False:
            distributions = self._encode(x)
            mu = distributions[:, :self.z_dim_tot]
            x_recon = self._decode(mu)
            x_recon = x_recon.view(x.size())
            return x_recon

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)  

class brnl_VAE(nn.Module):
    #https://davidstutz.de/bernoulli-variational-auto-encoder-in-torch/
    def __init__(self, z_dim=20,n_filter=32, nc=1, train=True):
        super(brnl_VAE, self).__init__()
        self.nc = nc
        self.z_dim_tot = z_dim
        self.z_dim_bern = z_dim
        self.n_filter = n_filter
        self.train = train
        #assume initial size is 32 x 32 
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

    def forward(self, x, current_flip_idx_norm=None, train=True ):
        self.train = train
        if self.train==True:
            p_dist = self._encode(x)
            p_dist = F.sigmoid(p_dist)
            if current_flip_idx_norm is not None:
                indx_vec = torch.zeros(p.size(0),1)
                ones = torch.ones(p.size(0),1)
                indx_vec = indx_vec + ones[current_flip_idx_norm]
                delta_mat = torch.zeros(p.size())
                    
                p[current_flip_idx_norm,1] = 1 - p[current_flip_idx_norm,1] 
            z = reparametrize_bernoulli(p_dist)
            x_recon = self._decode(z)
            x_recon = x_recon.view(x.size())
            return x_recon, p_dist
        elif self.train ==False:
            p_dist = self._encode(x)
            x_recon = self._decode(p_dist)
            x_recon = x_recon.view(x.size())
            return x_recon

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)  



class hybrid_VAE(nn.Module):
    def __init__(self, z_dim_bern=10, z_dim_gauss=10,n_filter=32, nc=1, train=True):
        super(hybrid_VAE, self).__init__()
        self.nc = nc
        self.z_dim_gauss = z_dim_gauss
        self.z_dim_bern = z_dim_bern
        self.z_dim_tot = z_dim_gauss + z_dim_bern
        self.train = train
        #assume initial size is 32 x 32 
        self.encoder = nn.Sequential(
            #nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            #nn.ReLU(True),
            nn.Conv2d(nc, n_filter, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(n_filter, n_filter, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(n_filter, n_filter, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, n_filter*4*4)),                  # B, 512
            nn.Linear(n_filter*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim_bern+2*z_dim_gauss),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim_tot, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, n_filter*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, n_filter, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(n_filter, n_filter, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(n_filter, n_filter, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(n_filter, nc, 4, 2, 1), # B,  32, 32, 32
            #nn.ReLU(True),
            #nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )
        self.weight_init() 
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, current_flip_idx_norm=None, train=True ):
        self.train = train
        if self.train==True:
            distributions = self._encode(x)
            p = distributions[:, :self.z_dim_bern]
            mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
            logvar = distributions[:, (self.z_dim_bern+self.z_dim_gauss):]
            #flip 1st z of all images that are inverted
            p = F.sigmoid(p)
            if current_flip_idx_norm is not None:
                delta_mat = torch.zeros(p.size())
                delta_mat[current_flip_idx_norm,1]=1
                p = p - 2*delta_mat*p + delta_mat
                #p[current_flip_idx_norm,1] = 1 - p[current_flip_idx_norm,1]
            #reparametrise
            bern_z = reparametrize_bernoulli(p)
            gaus_z = reparametrize_gaussian(mu, logvar)
            joint_z = torch.cat((bern_z,gaus_z), 1)
            x_recon = self._decode(joint_z)
            x_recon = x_recon.view(x.size())
            return x_recon, p, mu, logvar
        elif self.train ==False:
            distributions = self._encode(x)
            p = distributions[:, :self.z_dim_bern]
            mu = distributions[:,self.z_dim_bern:self.z_dim_gauss ]
            joint_z = torch.cat((p,mu), 1)
            #print(joint_z.size())
            x_recon = self._decode(joint_z)
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
   