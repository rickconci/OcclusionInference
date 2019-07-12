"""solver_mod.py"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import Image
import pickle
import math

import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision.utils import make_grid, save_image


from dataset_unsup import return_data_unsupervised, MyDataset_unsup
from model_BLT_VAE import BLT_mod, BLT_orig, FF
from visuals_mod import traverse_z, plotsave_tests, construct_z_hist



def reconstruction_loss(x, x_recon):
    #reconstruction loss for GAUSSIAN distribution of pixel values (not Bernoulli)
    batch_size = x.size(0)
    assert batch_size != 0
    x_recon = F.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size) #divide mse loss by batch size
    return recon_loss


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

    
def supervised_loss(output, target, encoder_target_type):
    batch_size = output.size(0)
    assert batch_size != 0
    
    if encoder_target_type =='joint':
        y_hat = output.float()
        y = target.float()
        y_hat = F.sigmoid(y_hat)
        e = 1e-20
        print(y[0,:].long())
        print(y_hat[0,:])
        sup_loss = (-torch.sum(y*torch.log(y_hat+e) + (1-y)*torch.log(1-y_hat+e))).div(batch_size)
        print(sup_loss)
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
        
        black_loss = F.cross_entropy(black_out, black_target, size_average=False).div(batch_size)
        white_loss = F.cross_entropy(white_out, white_target, size_average=False).div(batch_size)
        sup_loss = black_loss + white_loss
        
    elif encoder_target_type =='depth_black_white':
        depth = F.sigmoid(output[:,:1])
        black_out = output[:,1:11]
        white_out = output[:,11:]
        depth_target = target[:,:1]
        black_target = target[:,1:11]
        white_target = target[:,11:]
        depth_loss = F.binary_cross_entropy(depth, depth_target,size_average=False).div(batch_size)
        black_loss = F.cross_entropy(black, black_target, size_average=False).div(batch_size)
        white_loss = F.cross_entropy(white, white_target, size_average=False).div(batch_size)
        sup_loss = black_loss + white_loss + depth_loss

    return sup_loss




class Solver_unsup(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        
        self.model = args.model
        self.max_epoch = args.max_epoch
        self.global_iter = 0
        
        
        self.z_dim = args.z_dim
        self.bs = args.batch_size
        self.flip= args.flip
        self.testing_method = args.testing_method
        self.encoder_target_type = args.encoder_target_type
        
        if args.encoder_target_type == 'joint':
            self.z_dim = 10
        elif args.encoder_target_type == 'black_white':
            self.z_dim = 20
        elif args.encoder_target_type == 'depth_black_white':
            self.z_dim = 21
            
        if args.z_dim_bern is None and args.z_dim_gauss is None and args.model =='hybrid_VAE':
            self.z_dim_bern = math.floor(args.z_dim/2)
            self.z_dim_gauss = math.ceil(args.z_dim/2)
        else:
            self.z_dim_bern = args.z_dim_bern
            self.z_dim_gauss = args.z_dim_gauss
            
        self.n_filter = args.n_filter

        self.image_size = args.image_size
        self.beta = args.beta
        self.gamma = args.gamma
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        
        if args.dataset.lower() == 'digits_gray':
            self.nc = 1
        elif args.dataset.lower() == 'digits_col':
            self.nc = 3
        else:
            raise NotImplementedError
            
        if args.model == 'gauss_VAE':
            net = conv_VAE_32(z_dim=self.z_dim,n_filter=self.n_filter, nc=self.nc, train=True)
        elif args.model == 'brnl_VAE':
            net = brnl_VAE(z_dim=self.z_dim,n_filter=self.n_filter, nc=self.nc, train=True)
            self.beta = 0
        elif args.model =='hybrid_VAE':
            net = hybrid_VAE(z_dim_bern= self.z_dim_bern, z_dim_gauss =self.z_dim_gauss,
                             n_filter = self.n_filter, nc = self.nc, train=True)
        elif args.model =='BLT_encoder':
            net = BLT_encoder(z_dim=self.z_dim, nc=self.nc, batch_size=self.bs)
        else:
            raise NotImplementedError('Model not correct')
        
        print("CUDA availability: " + str(torch.cuda.is_available()))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        # copy the model to each device
        #self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.net = net.to(self.device) 
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        
 
        
        print("net on cuda: " + str(next(self.net.parameters()).is_cuda))
        
        #print parameters in model
        tot_size = 0
        for parameter in self.net.parameters():
            tot_size += parameter.numel()
        print(tot_size ,"parameters in the network!")
             
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)
            
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = args.output_dir #os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        
        
        if self.testing_method == 'supervised_encoder':
             self.train_dl, self.test_dl  = return_data_sup_encoder(args)
        elif self.testing_method == 'supervised_decoder':
            self.train_dl, self_test_dl, self.gnrl_dl = return_data_sup_decoder(args)
        elif self.testing_method =='unsupervised':
            self.train_dl, self.test_dl, self.gnrl_dl = return_data_unsup(args)
        #elif self.testing_method =='semisupervised':
        else:
            raise NotImplementedError
 
        self.gather = DataGather()
        
        if self.flip==True:
            self.flip_idx = pickle.load( open( "{}train_idx_to_flip.p".format(
                args.dset_dir), "rb" ) )
            self.flip_idx.sort()
            print(self.flip_idx[0:20])
            print(len(self.flip_idx), " flipped images!")
       
        
    def train(self):
        #self.net(train=True)
        iters_per_epoch = len(self.train_dl)
        print(iters_per_epoch)
        max_iter = self.max_epoch*iters_per_epoch
        batch_size = self.train_dl.batch_size
        current_idxs  = 0
        current_flip_idx = []
        count = 0

        out = False
        pbar = tqdm(total=max_iter)
        pbar.update(self.global_iter)
        
        while not out:
            for sample in self.train_dl:
                self.global_iter += 1
                pbar.update(1)
               
                if self.flip == True:
                    if count%iters_per_epoch==0:
                        print("RESETTING COUNTER")
                        count=0
                    current_idxs = range(count*batch_size, (count+1)*batch_size)
                    current_flip_idx = [x for x in self.flip_idx if x in current_idxs]
                    if not current_flip_idx:
                        current_flip_idx_norm = None
                    else:
                        current_flip_idx_norm = []
                        current_flip_idx_norm[:] = [i - count*batch_size for i in current_flip_idx]
                else:
                    current_flip_idx_norm = None
                
              
                    
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                
                if self.testing_method == 'supervised_encoder':
                    final_out, _, _ = self.net(x)
                    loss = supervised_loss(final_out, y, self.encoder_target_type)
                    l2 = 0
                    for p in self.net.parameters():
                        l2 = l2 + p.pow(2).sum()
                    print(0.0005 * l2)
                    loss = loss + 0.0005 * l2
                if self.testing_method =='supervised_decoder':
                    recon = self.net(x)
                    loss = supervised_decoder_loss(recon, y)
                if self.testing_method =='unsupervised':
                    if self.model == 'gauss_BLT_VAE':
                        x_recon, mu, logvar = self.net(x)
                        recon_loss = reconstruction_loss(y, x_recon)
                        total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(mu, logvar)
                        loss = recon_loss + self.beta*total_kld
                    elif self.model == 'brnl_BLT_VAE':
                        x_recon, p_dist = self.net(x)
                        recon_loss = reconstruction_loss(y, x_recon)
                        total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(p_dist)
                        loss = recon_loss + self.gamma *total_kld  
                    elif self.model == 'hybrid_BLT_VAE':
                        x_recon, p_dist, mu, logvar = self.net(x)
                        recon_loss = reconstruction_loss(y, x_recon)
                        total_kld_bern, dim_wise_kld_bern, mean_kld_bern = kl_divergence_bernoulli(p_dist)
                        total_kld_gauss, dim_wise_kld_gauss, mean_kld_gauss = kl_divergence_gaussian(
                            mu, logvar)
                        loss = recon_loss + self.gamma *total_kld_bern + self.beta*total_kld_gauss
                    
                
                self.adjust_learning_rate(self.optim, (count/iters_per_epoch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                count +=1 
                
                #if self.viz_on and self.global_iter%self.gather_step == 0:
                #    self.gather.insert(iter=self.global_iter,
                #                       p = p_dist.mean(0).data, 
                #                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                #                       recon_loss=recon_loss.data, total_kld_gauss=total_kld_gauss.data,
                #                       dim_wise_kld_gauss=dim_wise_kld_gauss.data,
                #                       mean_kld_gauss=mean_kld_gauss.data)
                    
                if self.global_iter%self.display_step == 0:
                    if self.model =='hybrid_VAE':
                        pbar.write('[{}] recon_loss:{:.3f} total_kld_gauss:{:.3f} mean_kld_gauss:{:.3f} total_kld_bern:{:.3f} mean_kld_bern:{:.3f}'.format(
                            self.global_iter, recon_loss.data,
                            total_kld_gauss.data[0], mean_kld_gauss.data[0], total_kld_bern.data[0],
                            mean_kld_bern.data[0]))
                    else:
                         pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} '.format(
                             self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0]))
                        
                    if self.model == 'gauss_VAE' or 'hybrid_VAE':
                        var = logvar.exp().mean(0).data
                        var_str = ''
                        for j, var_j in enumerate(var):
                            var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                        pbar.write(var_str)
                      
                    if self.viz_on:
                        print("now visdoming!")
                        self.gather.insert(images=y.data)
                        self.gather.insert(images=F.sigmoid(x_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.gather.flush()
                        
                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    self.test_loss()
                    self.gnrl_loss()
                    #if self.testLoss  < 
                    self.test_plots()
                    with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                        myfile.write('\n[{}] recon_loss:{:.3f}, test_loss:{:.3f}, gnrl_loss:{:.3f}'.format(
                            self.global_iter, recon_loss.data, self.testLoss, self.gnrlLoss))
                
                
                if self.global_iter%500 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    
    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (epoch / 40))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(lr)
    
    def run_model(self, model, x, y ):
        if model == 'gauss_VAE':
            x_recon, mu, logvar = self.net(x)
            recon_loss = reconstruction_loss(y, x_recon)
            total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(mu, logvar)
            vae_loss = recon_loss + self.beta*total_kld
            return([vae_loss,recon_loss, total_kld] )
        elif model == 'brnl_VAE':
            x_recon, p_dist = self.net(x)
            recon_loss = reconstruction_loss(y, x_recon)
            total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(p_dist)
            vae_loss = recon_loss + self.gamma *total_kld  
            return([vae_loss,recon_loss, total_kld] )
        elif model == 'hybrid_VAE':
            x_recon, p_dist, mu, logvar = self.net(x)
            recon_loss = reconstruction_loss(y, x_recon)
            total_kld_bern, dim_wise_kld_bern, mean_kld_bern = kl_divergence_bernoulli(p_dist)
            total_kld_gauss, dim_wise_kld_gauss, mean_kld_gauss = kl_divergence_gaussian(mu, logvar)
            vae_loss = recon_loss + self.gamma *total_kld_bern + self.beta*total_kld_gauss
            return([vae_loss,recon_loss, total_kld_bern, total_kld_gauss] ) 
        elif model =='BLT_encoder':
            final_out, _ , _= self.net(x)
            loss = supervised_loss(final_out, y, self.encoder_target_type)
            return([loss, final_out])
    
    def test_loss(self):
        print("Calculating test loss")
        testLoss = 0.0
        cnt = 0
        with torch.no_grad():
            for sample in self.test_dl:
                img = sample['x'].to(self.device)
                trgt = sample['y'].to(self.device)
                print(self.model)
                testLoss= self.run_model(self.model, img, trgt)
                if self.model =='BLT_encoder':
                    final_out =testLoss[1]
                    self.accuracy(final_out,trgt )
                testLoss = testLoss[0]
                cnt += 1
                print(cnt)
        self.testLoss = testLoss.div(cnt)
        self.testLoss = self.testLoss.cpu().numpy()#[0]
        print('[{}] test_Loss:{:.3f}'.format(self.global_iter, self.testLoss))
    
    def accuracy(self, outputs, targets):
        assert outputs.size() == targets.size()
        assert outputs.size(0) > 0
        x = torch.topk(outputs,2,dim=1 )
        y = torch.topk(targets,2,dim=1 )
        outputs = x[1].cpu().numpy()
        targets  = y[1].cpu().numpy()

        accuracy = np.sum(outputs == targets)/outputs.size *100
        print('[{}] accuracy:{:.3f}'.format(self.global_iter, accuracy))
        return(accuracy)
        

    
    
    
    def gnrl_loss(self):
        print("Calculating generalisation loss")
        gnrlLoss = 0.0
        cnt = 0
        with torch.no_grad():
            for sample in self.gnrl_data_loader:
                img = sample['x'].to(self.device)
                trgt = sample['y'].to(self.device)
                gnrlLoss= self.run_model(self.model, img, trgt)
                gnrlLoss = gnrlLoss[0]
                cnt += 1
                #print(cnt)
        self.gnrlLoss = gnrlLoss.div(cnt)
        self.gnrlLoss = self.gnrlLoss.numpy()[0]
        print('[{}] gnrl_Loss:{:.3f}'.format(self.global_iter, self.gnrlLoss))
        
        
    def test_plots(self):
        #self.net.eval()   but supposed to add when testing?
        net_copy = deepcopy(self.net)
        net_copy.to('cpu')
        
        print("creating sample images!")
        #Print sample images by decoding samples of normal distribution size of z_dim
        if self.model =='gauss_VAE':
            sample = torch.randn(16, self.z_dim)
        elif self.model == 'bern_VAE':
            sample = torch.rand(16, self.z_dim)
        elif self.model=='hybrid_VAE':
            sample_2 = torch.randn(16, self.z_dim_gauss)
            sample_1 = torch.rand(16, self.z_dim_bern)
            sample = torch.cat((sample_1,sample_2), 1)
        with torch.no_grad():
            test_recon = net_copy._decode(sample)
            torchvision.utils.save_image( F.sigmoid(test_recon).view(
                test_recon.size(0),1, self.image_size, self.image_size).data.cpu(), '{}/sampling_z_{}.png'.
                                         format(self.output_dir, self.global_iter))        
        
        #print("Constructing Z hist!")
        #construct_z_hist(net_copy, self.train_data_loader, self.global_iter, self.output_dir,dim='depth')

        
        #select test image to traverse 
        #print("Traversing!")
        test_image_paths = os.path.join(self.dset_dir + "test/orig/")
        test_target_paths = os.path.join(self.dset_dir + "test/inverse/")
        dset = MyDataset
        test_data = dset(test_image_paths,test_target_paths, image_size= self.image_size)
        #with torch.no_grad():
        #    for i in range(3):
        #        example_id = test_data.__getitem__(i)
        #        traverse_z(net_copy, example_id, ID=str(i),output_dir=self.output_dir, 
        #                   global_iter=self.global_iter, model= self.model, num_frames=200 )
    
        #create pdf with reconstructed test images 
        print('Reconstructing Test Images!')
        with torch.no_grad():
            plotsave_tests(net_copy, test_data, self.output_dir, self.global_iter,type="Test",  n=20 )
        
        print('Reconstructing generalisation images!')
        gnrl_image_paths = os.path.join(self.dset_dir + "gnrl/orig/")
        gnrl_target_paths = os.path.join(self.dset_dir + "gnrl/inverse/")
        dset = MyDataset
        grnl_data = dset(gnrl_image_paths,gnrl_target_paths, image_size= self.image_size)
        with torch.no_grad():
            plotsave_tests(net_copy, grnl_data, self.output_dir, self.global_iter, type="Gnrl", n=20 )

    
    def save_checkpoint(self, filename, silent=True):
        if torch.cuda.device_count()>1:
            model_states = {'net': self.net.module.state_dict(),}
        else:
            model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            if torch.cuda.device_count()>1:
                self.net.module.load_state_dict(checkpoint['model_states']['net'])
            else:
                self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

            
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

        