
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize_gaussian(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def reparametrize_bernoulli(p_dist):
    eps = Variable(p_dist.data.new(p_dist.size()).uniform_(0,1))
    z = F.sigmoid(torch.log(eps - 1e-20) - torch.log(1-eps+ 1e-20) + torch.log(
        p_dist- 1e-20) -torch.log(1-p_dist+ 1e-20))
    return z



class BLT_orig_encoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_orig_encoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using BLT_orig_encoder")
        
        self.W_b_1 = nn.Conv2d(1, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        self.W_b_2 = nn.Conv2d(32, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.Lin = nn.Linear(32, self.z_dim, bias=True)
        
        self.MPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_1.weight)
        nn.init.kaiming_uniform_(self.W_t_1.weight)        
        nn.init.kaiming_uniform_(self.W_b_2.weight)
        nn.init.kaiming_uniform_(self.W_l_2.weight)
        nn.init.kaiming_uniform_(self.Lin.weight)
        

    def forward(self, x):
        for t in range(4):
            if t<1:
                Z_1 = self.W_b_1(x)
                Z_2_mpool, indices_hid  = self.MPool(Z_1)
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_2_mpool)))
                read_out, indices_max =  F.max_pool2d_with_indices(self.LRN(F.relu(Z_2)), kernel_size=Z_2.size()[2:],
                                                               return_indices=True )
                final_z = self.Lin(read_out.view(-1, 32))
            if t >=1:
                Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                Z_2_mpool, indices_hid  = self.MPool(Z_1)
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_2_mpool))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                read_out, indices_max =  F.max_pool2d_with_indices(self.LRN(F.relu(Z_2)), kernel_size=Z_2.size()[2:],
                                                               return_indices=True )
                final_z = self.Lin(read_out.view(-1, 32))
    
        #print(torch.sum(torch.isnan(final_z)))
        #print(F.sigmoid(final_z[0,:]))
        #print(final_z.size())
        return(final_z)

class BLT_orig(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_orig, self).__init__()
        self.encoder = BLT_orig_encoder(z_dim, nc)
        
    def forward(self, x):
        z = self._encode(x)
        #recon = decoder(z)
        return(z)
    
    def _encode(self,x):
        return(self.encoder(x))
    
    #def _decode(self,z):
    #    return(elf.decoder(z))
        
class BLT_mod_encoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_mod_encoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using BLT_mod_encoder")
        
        self.W_b_1 = nn.Conv2d(1, 32, kernel_size= 4, stride = 2, padding = 1, bias=True)   # bs 32 16 16
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        
        self.W_b_2 = nn.Conv2d(32, 32, kernel_size= 4, stride = 2, padding = 1, bias=True) # bs 32 8 8
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        
        self.W_b_3 = nn.Conv2d(32, 32, kernel_size= 4, stride = 2, padding = 1, bias=True) # bs 32 4 4
        self.W_l_3 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        
        self.Lin_1 = nn.Linear(32*4*4, 256, bias=True)
        #if type == 'gaussian':
        self.Lin_2 = nn.Linear(256, self.z_dim*2, bias=True)
        #elif type == 'bernoulli':
        #    self.Lin_2 = nn.Linear(256, self.z_dim, bias=True)
        #elif type == 'hybrid':
        #    self.Lin_2 = nn.Linear(256, z_dim_brnl +2*z_dim_gauss, bias=True)
        
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_1.weight)
        nn.init.kaiming_uniform_(self.W_t_1.weight)        
        nn.init.kaiming_uniform_(self.W_b_2.weight)
        nn.init.kaiming_uniform_(self.W_l_2.weight)
        nn.init.kaiming_uniform_(self.W_t_2.weight)
        nn.init.kaiming_uniform_(self.W_b_3.weight)
        nn.init.kaiming_uniform_(self.W_l_3.weight)
        nn.init.kaiming_uniform_(self.Lin_1.weight )
        nn.init.kaiming_uniform_(self.Lin_2.weight )
        

    def forward(self, x):
        for t in range(4):
            if t <1:
                Z_1 = self.W_b_1(x)
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))
                read_z = self.Lin_1(Z_3.view(-1, 32*4*4 ))
                final_z = self.Lin_2(read_z)
            elif t>=1:
                Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) + self.W_t_2(self.LRN(F.relu(Z_3))) 
                Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3))) 
                read_z = self.Lin_1(Z_3.view(-1, 32*4*4 ))
                final_z = self.Lin_2(read_z)
                
        #print(torch.sum(torch.isnan(final_z)))
        #print(final_z.size())
        #print(final_z[0,:])
        return(final_z)


class BLT_mod_decoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_mod_decoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using BLT_mod_decoder")
        
        self.Lin_1 = nn.Linear(self.z_dim, 256, bias=True)
        self.Lin_2 = nn.Linear(256, 32*4*4, bias=True) 
        
        self.W_b_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=True ) # bs 32 8 8
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.Conv2d(32, 32, kernel_size= 4, stride = 2, padding = 1, bias=False)   
        
        self.W_b_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=True ) # bs 32 16 16
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_2 = nn.Conv2d(32, 32, kernel_size= 4, stride = 2, padding = 1, bias=False) 
        
        self.W_b_3 = nn.ConvTranspose2d(32, nc, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=True ) # bs 32 32 32
        self.W_l_3 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
                
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_1.weight)
        nn.init.kaiming_uniform_(self.W_t_1.weight)        
        nn.init.kaiming_uniform_(self.W_b_2.weight)
        nn.init.kaiming_uniform_(self.W_l_2.weight)
        nn.init.kaiming_uniform_(self.W_t_2.weight)
        nn.init.kaiming_uniform_(self.W_b_3.weight)
        nn.init.kaiming_uniform_(self.W_l_3.weight)
        nn.init.kaiming_uniform_(self.Lin_1.weight )
        nn.init.kaiming_uniform_(self.Lin_2.weight )
        
    def forward(self, z):
        for t in range(4):
            if t <1:
                Z_1 = self.Lin_2(self.Lin_1(z)).view(-1,32,4,4)
                Z_2 = self.W_b_1(Z_1)
                Z_3 = self.W_b_2(Z_2)
                final_img = self.W_b_3(Z_3)
            if t>=1:
                Z_1 = self.Lin_2(self.Lin_1(z)).view(-1,32,4,4) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2)))
                Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) +  self.W_t_2(self.LRN(F.relu(Z_3)))
                Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                final_img = self.W_b_3(Z_3)
        
        #print(torch.sum(torch.isnan(final_img)))
        #print(final_img.size())
        #print(final_img[0,:])
        return(final_img)

    
class BLT_mod(nn.Module):
    def __init__(self, z_dim, nc):
        super(BLT_mod, self).__init__()
        self.encoder = BLT_mod_encoder(z_dim, nc )
        self.decoder = BLT_mod_decoder(z_dim, nc )
        
    def forward(self, x):
        z = self._encode(x)
        recon = self._decode(z)
        recon = recon.view(x.size())
        return(recon)
    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))
    
class BLT_gauss_VAE(nn.Module):
    def __init__(self, z_dim_tot, nc):
        super(BLT_gauss_VAE, self).__init__()
        self.z_dim_tot = z_dim_tot
        self.encoder = BLT_mod_encoder(z_dim_tot, nc )
        self.decoder = BLT_mod_decoder(z_dim_tot, nc)
        print("using BLT_gauss_VAE")
    def forward(self, x,  train=True ):
        if train==True:
            distributions = self._encode(x)
            print(distributions.shape)
            mu = distributions[:, :self.z_dim_tot]
            logvar = distributions[:, self.z_dim_tot:]
            print(mu.shape)
            print(logvar.shape)
            z = reparametrize_gaussian(mu, logvar)
            x_recon = self._decode(z)
            x_recon = x_recon.view(x.size())
            return x_recon, mu, logvar
        elif train ==False:
            distributions = self._encode(x)
            mu = distributions[:, :self.z_dim_tot]
            x_recon = self._decode(mu)
            x_recon = x_recon.view(x.size())
            return x_recon
    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))
    
    
class BLT_brnl_VAE(nn.Module):
    def __init__(self, z_dim_tot, nc):
        super(BLT_brnl_VAE, self).__init__()
        self.z_dim_tot = z_dim_tot
        self.encoder = BLT_mod_encoder(z_dim_tot, nc )
        self.decoder = BLT_mod_decoder(z_dim_tot, nc)
        print('using BLT_brnl_VAE')
    def forward(self, x, current_flip_idx_norm=None, train=True ):
       
        if train==True:
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
        elif train ==False:
            p_dist = self._encode(x)
            x_recon = self._decode(p_dist)
            x_recon = x_recon.view(x.size())
            return x_recon
    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))  
    
    
class BLT_hybrid_VAE(nn.Module):
    def __init__(self, z_dim_bern, z_dim_gauss, nc):
        super(BLT_hybrid_VAE, self).__init__()
        self.z_dim_gauss = z_dim_gauss
        self.z_dim_bern = z_dim_bern
        self.z_dim_tot = z_dim_gauss + z_dim_bern
        self.encoder = BLT_mod_encoder(z_dim_tot, nc )
        self.decoder = BLT_mod_decoder(z_dim_tot, nc)
        print('using BLT_hybrid_VAE')
    def forward(self, x, current_flip_idx_norm=None, train=True ):
        if train==True:
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
            #reparametrise
            bern_z = reparametrize_bernoulli(p)
            gaus_z = reparametrize_gaussian(mu, logvar)
            joint_z = torch.cat((bern_z,gaus_z), 1)
            x_recon = self._decode(joint_z)
            x_recon = x_recon.view(x.size())
            return x_recon, p, mu, logvar
        elif train ==False:
            distributions = self._encode(x)
            p = distributions[:, :self.z_dim_bern]
            mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
            joint_z = torch.cat((p,mu), 1)
            x_recon = self._decode(joint_z)
            print(x_recon.shape)
            print(x_recon.shape)
            return x_recon

    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))  
    
class FF(nn.Module):
    def __init__(self, z_dim, nc=1, n_filter=32):
        super(FF, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.n_filter = n_filter
        print(print('using FF '))
        #assume initial size is 32 x 32 
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, self.n_filter, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.),
            nn.Conv2d(self.n_filter, self.n_filter, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.),
            nn.Conv2d(self.n_filter, self.n_filter, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.),
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
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.),
            nn.ConvTranspose2d(self.n_filter, self.n_filter, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.),
            nn.ConvTranspose2d(self.n_filter, nc, 4, 2, 1), # B,  32, 32, 32
        )
        self.weight_init() 
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x ):
            z = self._encode(x)
            x_recon = self._decode(z)
            x_recon = x_recon.view(x.size())
            return x_recon
     

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z) 
    
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

            
def spatial_broadcaster(Zs, h=32, w=32):
    z_b = np.tile(z, (h,w,1))
    z_b = torch.from_numpy(z_b).float()
    x = np.linspace(-1,1,w)
    y = np.linspace(-1,1,w)
    x_b, y_b =np.meshgrid(x,y)
    x_b = torch.from_numpy(x_b).float()
    x_b = torch.unsqueeze(x_b, 2)
    y_b = torch.from_numpy(y_b).float()
    y_b = torch.unsqueeze(y_b, 2)
    z_sb = torch.cat((z_b, x_b,y_b), -1)
    return(z_sb)