
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

def reparametrize_gaussian(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def reparametrize_bernoulli(p_dist):
    eps = Variable(p_dist.data.new(p_dist.size()).uniform_(0,1))
    z = F.sigmoid(torch.log(eps + 1e-20) - torch.log(1-eps+ 1e-20) + torch.log(p_dist + 1e-20) - torch.log(1-p_dist+ 1e-20))
    return z


##### UNSUPERVISED ######


class multi_encoder(nn.Module):
    def __init__(self, encoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p):
        super (multi_encoder, self).__init__()
        self.encoder_type = encoder_type
        self.n_rep = n_rep
        self.n_filt = n_filter
        
        
        self.W_b_1 = nn.Conv2d(nc, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)   # bs 32 16 16
        self.W_b_2 = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
        self.W_b_3 = nn.Conv2d(n_filter, n_filter, kernel_size= k, stride = 2, padding = p, bias=True)
        
        if encoder_type == 'BL' or encoder_type == 'BLT':
            self.W_l_1 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_2 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_3 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
        if encoder_type == 'BT' or encoder_type == 'BLT':
            self.W_t_1 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
            self.W_t_2 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=4, stride=2, padding=1, output_padding=0 ,bias=False )
        
        
        self.Lin_1 = nn.Linear(n_filter*4*4, 256, bias=True)
        self.Lin_2 = nn.Linear(256, 256, bias=True)
        self.Lin_3 = nn.Linear(256, (z_dim_bern+2*z_dim_gauss), bias=True)
        
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        self.weight_init()
    
    def weight_init(self):
        for block in self._modules.values():
            #print(block)
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)
    
    def forward(self, x):
        final_z_list = []
        
        if self.encoder_type == 'B':
            Z_1 = self.W_b_1(x)
            Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
            Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))
            final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
        elif self.encoder_type == 'BL':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) 
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1)))
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
        elif self.encoder_type == 'BT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))   
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_t_2(self.LRN(F.relu(Z_3)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
        elif self.encoder_type == 'BLT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.W_b_1(x)
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1)))
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) 
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
                elif t>=1:
                    Z_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1))) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_2(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) + self.W_t_2(self.LRN(F.relu(Z_3))) 
                    Z_3 = self.W_b_3(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
            
        
        #final_z_list.append(self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(self.LRN(F.relu(Z_3)).view(-1, self.n_filt*4*4 )))))))
           
                
        #print(torch.sum(torch.isnan(final_z)))
        #print(final_z.size())
        #print(F.sigmoid(final_z[0,:]))
        return(final_z_list)

    
class multi_decoder(nn.Module):
    def __init__(self, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p):
        super (multi_decoder, self).__init__()    
        self.decoder_type = decoder_type
        self.n_rep = n_rep
        self.n_filter = n_filter
        
        self.Lin_1 = nn.Linear( z_dim_bern + z_dim_gauss, 256, bias=True)
        self.Lin_2 = nn.Linear(256, 256, bias=True) 
        self.Lin_3 = nn.Linear(256, n_filter*4*4, bias=True)
        
        self.W_b_1 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
        self.W_b_2 = nn.ConvTranspose2d(n_filter, n_filter, kernel_size=k, stride=2, padding=p, bias=True )
        self.W_b_3 = nn.ConvTranspose2d(n_filter, nc, kernel_size=k, stride=2, padding=p, bias=True ) 
        
        if decoder_type == 'BL' or decoder_type == 'BLT':
            self.W_l_1 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_2 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
            self.W_l_3 = nn.Conv2d(n_filter, n_filter,kernel_size= 3, stride = 1, padding = 1, bias=False)
        if decoder_type == 'BT' or decoder_type == 'BLT':
            self.W_t_1 = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False)
            self.W_t_2 = nn.Conv2d(n_filter, n_filter, kernel_size= 4, stride = 2, padding = 1, bias=False) 
            
                
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
      
        self.weight_init()
        
    def weight_init(self):
        for block in self._modules.values():
            #print(block)
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)
            
                    
    def forward(self, z):
        final_img_list = []
        
        if self.decoder_type == 'B':
            Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
            Z_2 = self.W_b_1(F.relu(Z_1))
            Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
            final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
        
        elif self.decoder_type == 'BL':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_l_1(self.LRN(F.relu(Z_1))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
        elif self.decoder_type =='BT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_t_1(self.LRN(F.relu(Z_2))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) +  self.W_t_2(self.LRN(F.relu(Z_3))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) 
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
        elif self.decoder_type == 'BLT':
            for t in range(self.n_rep):
                if t <1:
                    Z_1 = self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4)
                    Z_2 = self.W_b_1(F.relu(Z_1))
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
                if t>=1:
                    Z_1 =  self.Lin_3(F.relu(self.Lin_2(F.relu(self.Lin_1(z))))).view(-1,self.n_filter,4,4) + self.W_t_1(self.LRN(F.relu(Z_2))) + self.W_l_1(self.LRN(F.relu(Z_1))) 
                    Z_2 = self.W_b_1(self.LRN(F.relu(Z_1))) +  self.W_t_2(self.LRN(F.relu(Z_3))) + self.W_l_2(self.LRN(F.relu(Z_2))) 
                    Z_3 = self.W_b_2(self.LRN(F.relu(Z_2))) + self.W_l_3(self.LRN(F.relu(Z_3)))
                    final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
            
                
        #final_img_list.append(self.W_b_3(self.LRN(F.relu(Z_3))))
        #final_img = self.W_b_3(self.LRN(F.relu(Z_3)))
        return final_img_list
        
        
class multi_VAE(nn.Module):
    def __init__(self, encoder_type, decoder_type, z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, sbd, k, p, AE):
        super (multi_VAE, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.z_dim_bern = z_dim_bern
        self.z_dim_gauss = z_dim_gauss
        self.z_dim_tot = z_dim_bern + z_dim_gauss
        self.nc = nc
        self.n_filter = n_filter
        self.sbd = sbd
        self.AE = AE
        if self.AE:
            print("NORMAL AUTOENCODER")
        
        self.encoder = multi_encoder(encoder_type,z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p)
        if sbd == True:
            self.decoder = SB_decoder(z_dim_bern, z_dim_gauss, n_filter, nc)
            self.sbd_model = spatial_broadcast_decoder()
            print('using {} encoder SBD decoder'.format(encoder_type))
        else:
            self.decoder = multi_decoder(decoder_type,z_dim_bern, z_dim_gauss, n_filter, nc, n_rep, k, p)
            print('using {} encoder {} decoder'.format(encoder_type, decoder_type))

    def forward(self, x, current_flip_idx_norm=None, train=True ):
        if train==True:
            if not self.AE:
                distributions = self._encode(x)
                distributions = distributions[-1]
                if self.z_dim_bern == 0:
                    p = 0
                    mu = distributions[:,:self.z_dim_gauss]
                    logvar = distributions[:, self.z_dim_gauss: ]
                    z = reparametrize_gaussian(mu, logvar)
                elif self.z_dim_gauss == 0:
                    mu, logvar = 0
                    p = distributions[:, :self.z_dim_bern]
                    p = F.sigmoid(p)
                    if current_flip_idx_norm is not None:
                        delta_mat = torch.zeros(p.size())
                        delta_mat[current_flip_idx_norm,1]=1
                        p = p - 2*delta_mat*p + delta_mat
                    z =  reparametrize_bernoulli(p)
                elif self.z_dim_bern !=0 and self.z_dim_gauss != 0:
                    p = distributions[:, :self.z_dim_bern]
                    p = F.sigmoid(p)
                    if current_flip_idx_norm is not None:
                        delta_mat = torch.zeros(p.size())
                        delta_mat[current_flip_idx_norm,1]=1
                        p = p - 2*delta_mat*p + delta_mat
                    mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
                    logvar = distributions[:, (self.z_dim_bern+self.z_dim_gauss):]
                    bern_z =reparametrize_bernoulli(p)
                    gauss_z = reparametrize_gaussian(mu, logvar)
                    z = torch.cat((bern_z,gauss_z), 1)

                if self.sbd:
                    z = self.sbd_model(z)
            elif self.AE:
                
                z = self._encode(x)

                #print("z_max", torch.max(z[-1]))
                #print("z_min", torch.min(z[-1]))
                
                p = torch.tensor(0); mu = torch.tensor(0); logvar = torch.tensor(0)
                #print("No reparam!")
                
            x_recon_list = self._decode(z[-1])
            #x_recon_list = [m.view(x.size()) for m in x_recon_list]
            return x_recon_list, p, mu, logvar
        
        elif train ==False:
            distributions_list = self._encode(x)
            distributions = distributions_list[-1]
            p = distributions[:, :self.z_dim_bern]
            mu = distributions[:,self.z_dim_bern:(self.z_dim_bern+self.z_dim_gauss) ]
            z = torch.cat((p,mu), 1)
            if self.sbd:
                z = self.sbd_model(z)
            x_recon = self._decode(z)
            #print(x_recon.shape)
            return x_recon

    
    def _encode(self,x):
        return(self.encoder(x))
    
    def _decode(self,z):
        return(self.decoder(z))  

     
    
class SB_decoder(nn.Module):
    def __init__(self, z_dim_bern, z_dim_gauss, n_filter, nc):
        super(SB_decoder, self).__init__()  
            
        self.decoder = nn.Sequential(
            nn.Conv2d((z_dim_bern + z_dim_gauss + 2), n_filter, 3, 1, 1),     
            nn.ReLU(True),
            nn.Conv2d(n_filter, n_filter, 3, 1, 1),         
            nn.ReLU(True),
            nn.Conv2d(n_filter, nc, 3, 1, 1),         
        )
        self.weight_init() 
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        recon = self._decode(z)
        return(recon)
       
    def _decode(self,z):
        return(self.decoder(z))

    
class spatial_broadcast_decoder(nn.Module):
    def __init__(self, im_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(spatial_broadcast_decoder, self).__init__()
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.to(self.device)
        y_grid = y_grid.to(self.device)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
    
    def forward(self,z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))
        # Tile across to match image size
        # Shape: NxDx32x32
        z = z.expand(-1, -1, self.im_size, self.im_size)
        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x32x32
        z_bd = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                        self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        return(z_bd)
    
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

    
    
    
    
    
    
    
    
###### SUPERVISED ######

            
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
        
        self.weight_init()
        
    def weight_init(self):
        for block in self._modules.values():
            if isinstance(block, (nn.Linear, nn.Conv2d)):
                print(block)
                init.kaiming_normal(block.weight,  nonlinearity='relu')
                if block.bias is not None:
                    block.bias.data.fill_(0)

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
    

    