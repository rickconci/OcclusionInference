
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class BLT_encoder(nn.Module):
    def __init__(self, z_dim=20, nc=1, batch_size=200):
        super(BLT_encoder, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.bs =  batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.W_b_1 = nn.Conv2d(1, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        self.W_b_2 = nn.Conv2d(32, 32, kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.Lin = nn.Linear(32, self.z_dim, bias=False)
        
        self.MPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_1.weight)
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_b_2.weight )
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_2.weight)
        nn.init.kaiming_uniform_(self.Lin.weight )
        

    def forward(self, x):
        Z_1_0 = torch.zeros(self.bs, 32, 32, 32, requires_grad=True).to(self.device)
        Z_1_1 = torch.zeros(self.bs, 32, 32, 32, requires_grad=True).to(self.device)
        Z_1_2 = torch.zeros(self.bs, 32, 32, 32, requires_grad=True).to(self.device)
        Z_1_3 = torch.zeros(self.bs, 32, 32, 32, requires_grad=True).to(self.device)
        Z_2_0 = torch.zeros(self.bs, 32,16, 16, requires_grad=True).to(self.device)
        Z_2_1 = torch.zeros(self.bs, 32,16, 16, requires_grad=True).to(self.device)
        Z_2_2 = torch.zeros(self.bs, 32,16, 16, requires_grad=True).to(self.device)
        Z_2_3 = torch.zeros(self.bs, 32,16, 16, requires_grad=True).to(self.device)
        read_out_0 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        read_out_1 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        read_out_2 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        final_z_0 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        final_z_1 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        final_z_2 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True).to(self.device)
        
        #t=0
        Z_1_0 = self.W_b_1(x)
        Z_2_mpool_0, indices_hid  = self.MPool(Z_1_0)
        Z_2_0 = self.W_b_2(self.LRN(F.relu(Z_2_mpool_0)))
        read_out_0, indices_max =  F.max_pool2d_with_indices(Z_2_0, kernel_size=Z_2_0.size()[2:],
                                                               return_indices=True )
        final_z_0 = self.Lin(read_out_0.view(-1, 32))
        #t=1
        Z_1_1 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1_0))) + self.W_t_1(self.LRN(F.relu(Z_2_0))) 
        Z_2_mpool_1, indices_hid  = self.MPool(Z_1_1)
        Z_2_1 = self.W_b_2(self.LRN(F.relu(Z_2_mpool_1))) + self.W_l_2(self.LRN(F.relu(Z_2_0))) 
        read_out_1, indices_max =  F.max_pool2d_with_indices(Z_2_1, kernel_size=Z_2_1.size()[2:],
                                                               return_indices=True )
        final_z_1 = self.Lin(read_out_1.view(-1, 32))
        #t=2
        Z_1_2 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1_1))) + self.W_t_1(self.LRN(F.relu(Z_2_1)))
        Z_2_mpool_2, indices_hid  = self.MPool(Z_1_2)
        Z_2_2 = self.W_b_2(self.LRN(F.relu(Z_2_mpool_2))) + self.W_l_2(self.LRN(F.relu(Z_2_1))) 
        read_out_2, indices_max =  F.max_pool2d_with_indices(Z_2_2, kernel_size=Z_2_2.size()[2:],
                                                               return_indices=True )
        final_z_2 = self.Lin(read_out_2.view(-1, 32))
        #t=3
        Z_1_3 = self.W_b_1(x) + self.W_l_1(self.LRN(F.relu(Z_1_2))) + self.W_t_1(self.LRN(F.relu(Z_2_2)))
        Z_2_mpool_3, indices_hid  = self.MPool(Z_1_3)
        Z_2_3 = self.W_b_2(self.LRN(F.relu(Z_2_mpool_3))) + self.W_l_2(self.LRN(F.relu(Z_2_2))) 
        read_out_3, indices_max =  F.max_pool2d_with_indices(Z_2_3, kernel_size=Z_2_3.size()[2:],
                                                               return_indices=True )
        final_z_3 = self.Lin(read_out_3.view(-1, 32))
           
            #print(final_z)
        return(final_z_3, indices_hid, indices_max)

    
class BLT_decoder(nn.Module):
    def __init__(self, h=32,w=32, nc=1, batch_size=200):
        super(BLT_encoder, self).__init__()
        self.nc = nc
        self.h = h
        self.w = w
        self.bs =  batch_size
        
        self.W_b_1 = nn.Conv2d(1, 32, kernel_size= 3, stride = 1, padding = 1, bias=True)
        self.W_l_1 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_t_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1 ,bias=False )
        self.W_b_2 = nn.Conv2d(32, 32, kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.W_l_2 = nn.Conv2d(32, 32,kernel_size= 3, stride = 1, padding = 1, bias=False)
        self.Lin = nn.Linear(32, self.z_dim, bias=False)
        
        self.MPool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.LRN = nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.5, k=1.)
        
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_1.weight)
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_b_2.weight )
        nn.init.kaiming_uniform_(self.W_b_1.weight)
        nn.init.kaiming_uniform_(self.W_l_2.weight)
        nn.init.kaiming_uniform_(self.Lin.weight )
        

    def forward(self, z, indices_max, indices_hid):
        Z_1 = torch.zeros(self.bs, 32, 32, 32, requires_grad=True)
        Z_2 = torch.zeros(self.bs, 32,16, 16, requires_grad=True)
        read_out_1 = torch.zeros(self.bs, 32, 1, 1 , requires_grad=True)
        final_z = torch.zeros(self.bs, self.z_dim,requires_grad=True)
        
        
        for t in range(4):
            print("TIME :" ,t)
            #print('Z1 %0s:', (Z_1 == 0).sum()/Z_1.numel()*100)
            #print('Z2 %0s:', (Z_2 == 0).sum()/Z_2.numel()*100)
            #print('read_out 0s:', (read_out_1 == 0).sum()/read_out_1.numel()*100)
            Z_1 = F.max_unpool2d(z, indices_max, kernel_size=2, stride=2, output_size=(self.bs, self.nc, self.h/2, self.w/2))
            Z_1 = Z_1 + self.W_l_1(self.LRN(F.relu(Z_1))) 
            Z_1 = Z_1 + self.W_t_1(self.LRN(F.relu(Z_2))) 
            Z_2_mpool, idx  = self.MPool(self.LRN(F.relu(Z_1)))
            Z_2 = Z_2 + self.W_b_2(Z_2_mpool)
            Z_2 = Z_2 + self.W_l_2(self.LRN(F.relu(Z_2))) 
            print(Z_2.shape)
            read_out_1 = read_out_1+ F.max_pool2d(Z_2, kernel_size=Z_2.size()[2:])
           
            final_z = final_z + self.Lin(read_out_1.view(-1, 32))
           
            print(final_z)
        return final_z, idx


    
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