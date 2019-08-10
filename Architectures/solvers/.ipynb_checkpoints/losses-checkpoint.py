import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision.utils import make_grid, save_image

def kl_divergence_gaussian(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def kl_divergence_bernoulli(p):
    batch_size = p.size(0)
    assert batch_size != 0
    if p.data.ndimension() == 4:
        p = p.view(p.size(0), p.size(1))
    
    prior= torch.tensor(0.5)
    klds = torch.mul(p, torch.log(p + 1e-20) - torch.log(prior)) + torch.mul(
        1 - p, torch.log(1 - p + 1e-20) - torch.log(1 - prior))    
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld


def reconstruction_loss(x, x_recon):
    #reconstruction loss for GAUSSIAN distribution of pixel values (not Bernoulli)
    batch_size = x.size(0)
    assert batch_size != 0
    
    x_recon = F.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size) #divide mse loss by batch size
    return recon_loss



def supervised_encoder_loss(output, target, n_digts, encoder_target_type):
    batch_size = output.size(0)
    assert batch_size != 0
    
    if encoder_target_type =='joint':
        y_hat = output.float()
        y = target.float()
        y_hat = F.sigmoid(y_hat)
        e = 1e-20
        #print(y[0,:].long())
        #print(y_hat[0,:])
        sup_loss = (-torch.sum(y*torch.log(y_hat-e) + (1-y)*torch.log(1-y_hat+e))).div(batch_size)
        #print(sup_loss)
        return sup_loss
    elif encoder_target_type =='black_white':
        assert output.size() == target.size()
        assert output.size(1) == 20
        output = output.float()
        target = target.long()
        black_out = output[:,:10]
        white_out = output[:,10:]
        black_target = torch.topk(target[:,:10],1,dim=1 )[1]
        white_target = torch.topk(target[:,10:], 1,dim=1 )[1]
        black_target = torch.squeeze(black_target)
        white_target = torch.squeeze(white_target)
        #zzz, idx = torch.max(F.softmax(black_out[0,:]) , 0)
        #print( idx ,black_target[0] )
        
        black_loss = F.cross_entropy(black_out, black_target, size_average=False).div(
            batch_size)
        white_loss = F.cross_entropy(white_out, white_target, size_average=False).div(batch_size)
        sup_loss = black_loss + white_loss
        return sup_loss
        
    elif encoder_target_type =='depth_black_white':
        depth = F.sigmoid(output[:,:1])
        black_out = output[:,1:11]
        white_out = output[:,11:]
        depth_target = target[:,:1]
        black_target = torch.topk(target[:,1:11],1,dim=1 )[1].long()
        white_target = torch.topk(target[:,11:],1,dim=1 )[1].long()
        depth_loss = F.binary_cross_entropy(depth, depth_target,size_average=False).div(batch_size)
        black_loss = F.cross_entropy(black, black_target, size_average=False).div(batch_size)
        white_loss = F.cross_entropy(white, white_target, size_average=False).div(batch_size)
        sup_loss = black_loss + white_loss + depth_loss
        return([sup_loss,depth_loss, black_loss, white_loss, 0])
    
    elif encoder_target_type =='depth_black_white_xy_xy':
        depth = F.sigmoid(output[:,:1])
        black_out = output[:,1:11]
        white_out = output[:,11:21]
        xy_xy = output[:,21:]
        depth_target = target[:,:1].float()
        black_target = torch.topk(target[:,1:11],1,dim=1)[1].squeeze(1)
        white_target = torch.topk(target[:,11:21],1,dim=1)[1].squeeze(1)
        xy_xy_target = target[:,21:].float()
    
        depth_loss = F.binary_cross_entropy(depth, depth_target,size_average=False).div(batch_size)
        black_loss = F.cross_entropy(black_out, black_target, size_average=False).div(batch_size)
        white_loss = F.cross_entropy(white_out, white_target, size_average=False).div(batch_size)
        xy_xy_loss = F.mse_loss(xy_xy, xy_xy_target, size_average=False).div(batch_size)
        
        sup_loss = black_loss + white_loss + depth_loss + xy_xy_loss
        return([sup_loss,depth_loss, black_loss, white_loss,xy_xy_loss])
    
    elif encoder_target_type== "depth_ordered_one_hot": 
        if n_digts ==2:
            back_out = output[:,0:10]
            front_out = output[:,10:]
            back_target = torch.topk(target[:,0:10],1,dim=1)[1].squeeze(1)
            front_target = torch.topk(target[:,10:],1,dim=1)[1].squeeze(1)
            back_loss = F.cross_entropy(back_out, back_target, size_average=False).div(batch_size)
            front_loss = F.cross_entropy(front_out, front_target, size_average=False).div(batch_size)
            sup_loss = back_loss + front_loss
            return([sup_loss, back_loss, front_loss, 0])
        
        elif n_digts ==3:
            back_out = output[:,0:10]
            mid_out = output[:,10:20]
            front_out = output[:,20:]
            back_target = torch.topk(target[:,0:10],1,dim=1)[1].squeeze(1)
            mid_target = torch.topk(target[:,10:20],1,dim=1)[1].squeeze(1)
            front_target = torch.topk(target[:,20:],1,dim=1)[1].squeeze(1)
            back_loss = F.cross_entropy(back_out, back_target, size_average=False).div(batch_size)
            mid_loss = F.cross_entropy(mid_out, mid_target, size_average=False).div(batch_size)
            front_loss = F.cross_entropy(front_out, front_target, size_average=False).div(batch_size)
            sup_loss = back_loss + mid_loss+ front_loss
            return([sup_loss, back_loss, mid_loss, front_loss, 0])
            
    elif encoder_target_type== "depth_ordered_one_hot_xy" :
        if n_digts ==2:
            back_out = output[:,0:10]
            front_out = output[:,10:20]
            back_target = torch.topk(target[:,0:10],1,dim=1)[1].squeeze(1)
            front_target = torch.topk(target[:,10:20],1,dim=1)[1].squeeze(1)
            xy_out =  output[:,20:]
            xy_target = target[:,20:]
            back_loss = F.cross_entropy(back_out, back_target, size_average=False).div(batch_size)
            front_loss = F.cross_entropy(front_out, front_target, size_average=False).div(batch_size)
            xy_loss =  F.mse_loss(xy_out, xy_target, size_average=False).div(batch_size)
            sup_loss = back_loss + front_loss + xy_loss
       
            return([sup_loss, back_loss, torch.tensor(0), front_loss, xy_loss])
        
        elif n_digts ==3:
            back_out = output[:,0:10]
            mid_out = output[:,10:20]
            front_out = output[:,20:30]
            back_target = torch.topk(target[:,0:10],1,dim=1)[1].squeeze(1)
            mid_target = torch.topk(target[:,10:20],1,dim=1)[1].squeeze(1)
            front_target = torch.topk(target[:,20:30],1,dim=1)[1].squeeze(1)
            xy_out =  output[:,30:]
            xy_target = target[:,30:]
            back_loss = F.cross_entropy(back_out, back_target, size_average=False).div(batch_size)
            mid_loss = F.cross_entropy(mid_out, mid_target, size_average=False).div(batch_size)
            front_loss = F.cross_entropy(front_out, front_target, size_average=False).div(batch_size)
            xy_loss =  F.mse_loss(xy_out, xy_target, size_average=False).div(batch_size)
            sup_loss = back_loss + mid_loss+ front_loss + xy_loss
         
            return([sup_loss, back_loss, mid_loss, front_loss, xy_loss])
        
                            
    

def supervised_decoder_loss(img, recon):
    #reconstruction loss for GAUSSIAN distribution of pixel values (not Bernoulli)
    batch_size = recon.size(0)
    assert batch_size != 0
    recon = F.sigmoid(recon)
    recon_loss = F.mse_loss(recon, img, size_average=False).div(batch_size) #divide mse loss by batch size
    return recon_loss