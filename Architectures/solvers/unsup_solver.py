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

import sys
sys.path.insert(0, '/Users/riccardoconci/Desktop/code/ZuckermanProject/OcclusionInference/Architectures')
#sys.path.insert(0, '/home/riccardo/Desktop/OcclusionInference/Architectures')

from data_loaders.dataset_unsup import return_data_unsupervised
from models.BLT_models import multi_VAE, SB_decoder, spatial_broadcast_decoder

from solvers.visuals_mod import traverse_z,construct_z_hist, plotsave_tests
from solvers.losses import reconstruction_loss, kl_divergence_gaussian, kl_divergence_bernoulli
from solvers.utils_mod import DataGather            

class Solver_unsup(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.args = args
        
        self.encoder = args.encoder
        self.decoder = args.decoder
        self.z_dim_bern = args.z_dim_bern
        self.z_dim_gauss= args.z_dim_gauss
        
        self.n_filter = args.n_filter
        self.n_rep = args.n_rep
        self.kernel_size = args.kernel_size
        self.padding = args.padding
        self.sbd = args.sbd
        
        if args.dataset.lower() == 'digits_gray':
            self.nc = 1
        elif args.dataset.lower() == 'digits_col':
            self.nc = 3
        else:
            raise NotImplementedError
        
        net = multi_VAE(self.encoder,self.decoder,self.z_dim_bern,self.z_dim_gauss,self.n_filter,self.nc,
                        self.n_rep,self.sbd, self.kernel_size, self.padding)
        
        
        #print parameters in model
        encoder_size = 0
        decoder_size = 0
        for name, param in net.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_size += param.numel()
                elif 'decoder' in name:
                    decoder_size += param.numel()
        tot_size = encoder_size + decoder_size
        print(tot_size ,"parameters in the network!", 
              encoder_size, " in the encoder",
             decoder_size, "in the decoder")
        self.params = tot_size
        
        print("CUDA availability: " + str(torch.cuda.is_available()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        self.net = net.to(self.device) 
        print("net on cuda: " + str(next(self.net.parameters()).is_cuda))
        
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        if args.optim_type =='Adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif args.optim_type =='SGD':
            self.optim = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
   
        self.train_dl, self.test_dl, self.gnrl_dl, self.test_data, self.gnrl_data = return_data_unsupervised(args)
        #
        
        self.max_epoch = args.max_epoch
        self.global_iter = 0
        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step
        
        self.beta = args.beta
        self.gamma = args.gamma
        self.l2_loss = args.l2_loss
        self.encoder_target_type = args.encoder_target_type
        self.image_size = args.image_size

        self.flip= args.flip
        if self.flip==True:
            self.flip_idx = pickle.load( open( "{}train_idx_to_flip.p".format(
                args.dset_dir), "rb" ) )
            self.flip_idx.sort()
            print(self.flip_idx[0:20])
            print(len(self.flip_idx), " flipped images!")
        
        self.testing_method = args.testing_method
        self.gather = DataGather(self.testing_method, self.encoder_target_type)
        
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        #if self.viz_on:
        #    self.viz = visdom.Visdom(port=self.viz_port)
        
        self.save_output = args.save_output
        self.output_dir = args.output_dir #os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.ckpt_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)
            
            
    def train(self):
        #self.net(train=True)
        iters_per_epoch = len(self.train_dl)
        print(iters_per_epoch, 'iters per epoch')
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
                
                x_recon, p_dist, mu, logvar = self.net(x, train=True)
                
                recon_loss = reconstruction_loss(y, x_recon)
                
                if self.z_dim_bern == 0:                    
                    total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(mu, logvar)
                    KL_loss = self.beta*total_kld
                elif self.z_dim_gauss == 0:
                    total_kld, dim_wise_kld, mean_kld = kl_divergence_bernoulli(p_dist)
                    KL_loss = self.gamma *total_kld 
                elif self.z_dim_bern !=0 and self.z_dim_gauss != 0:
                    total_kld_bern, dim_wise_kld_bern, mean_kld_bern = kl_divergence_bernoulli(p_dist)
                    total_kld_gauss, dim_wise_kld_gauss, mean_kld_gauss = kl_divergence_gaussian(mu, logvar)
                    KL_loss = self.gamma *total_kld_bern + self.beta*total_kld_gauss
                    
                loss = recon_loss + KL_loss

                self.adjust_learning_rate(self.optim, (count/iters_per_epoch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                count +=1 
                
                if self.global_iter%self.gather_step == 0:
                    self.test_loss()
                    if self.gnrl_dl != 0:
                        self.gnrl_loss()
                        with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                            myfile.write('\n[{}] train_loss:{:.3f},  train_recon_loss:{:.3f}, train_KL_loss:{:.3f}, test_loss:{:.3f}, test_recon_loss:{:.3f} , test_KL_loss:{:.3f}, gnrl_loss:{:.3f}, gnrl_recon_loss:{:.3f}, gnrl_KL_loss:{:.3f}'.format(self.global_iter, float(loss.data), float(recon_loss.data), float(KL_loss.data), self.testLoss, self.test_recon_loss, self.test_kl_loss, self.grnlLoss, self.gnrl_recon_loss, self.gnrl_kl_loss))
                            
                        self.gather.insert(iter=self.global_iter, trainLoss = loss.data, 
                                           train_recon_loss=recon_loss.data, train_KL_loss =KL_loss.data,
                                           testLoss = self.testLoss, test_recon_loss =self.test_recon_loss,
                                          test_kl_loss = self.test_kl_loss, grnlLoss = self.grnlLoss,
                                           gnrl_recon_loss =  self.gnrl_recon_loss,gnrl_kl_loss = self.gnrl_kl_loss  )
                    else:
                        with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                            myfile.write('\n[{}] train_loss:{:.3f},  train_recon_loss:{:.3f}, train_KL_loss:{:.3f}, test_loss:{:.3f}, test_recon_loss:{:.3f} , test_KL_loss:{:.3f}'.format(self.global_iter, float(loss.data), float(recon_loss.data), float(KL_loss.data), self.testLoss, self.test_recon_loss, self.test_kl_loss,))
                             
                        self.gather.insert(iter=self.global_iter, trainLoss = loss.data, 
                                    train_recon_loss=recon_loss.data, train_KL_loss =KL_loss.data,
                                    testLoss = self.testLoss, test_recon_loss =self.test_recon_loss,
                                        test_kl_loss = self.test_kl_loss )
                
                if self.global_iter%self.display_step == 0:
                    if self.z_dim_bern !=0 and self.z_dim_gauss != 0:
                        pbar.write('[{}] recon_loss:{:.3f} total_kld_gauss:{:.3f} mean_kld_gauss:{:.3f} total_kld_bern:{:.3f} mean_kld_bern:{:.3f}'.format(
                            self.global_iter, recon_loss.data,
                            total_kld_gauss.data[0], mean_kld_gauss.data[0], total_kld_bern.data[0],
                            mean_kld_bern.data[0]))
                    else:
                         pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} '.format(
                             self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0]))
                        
                    if self.z_dim_bern !=0:
                        var = logvar.exp().mean(0).data
                        var_str = ''
                        for j, var_j in enumerate(var):
                            var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                        pbar.write(var_str)
                      
                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last') 
                    oldtestLoss = self.testLoss
                    self.test_loss()
                    if self.gnrl_dl != 0:
                        self.gnrl_loss()
                    print('old test loss', oldtestLoss,'current test loss', self.testLoss )
                    if self.testLoss  < oldtestLoss:
                        self.save_checkpoint('best')
                        pbar.write('Saved best checkpoint(iter:{})'.format(self.global_iter))
                    
                    self.test_plots()
                    self.gather.save_data(self.global_iter, self.output_dir, 'last' )
                    
                if self.global_iter%500 == 0:
                    self.save_checkpoint(str(self.global_iter))
                    self.gather.save_data(self.global_iter, self.output_dir, None )

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
    
    def run_model(self, x, y):
        x_recon, p_dist, mu, logvar = self.net(x, train=True)
        recon_loss = reconstruction_loss(y, x_recon)
        if self.z_dim_bern == 0:                    
            total_kld, dim_wise_kld, mean_kld = kl_divergence_gaussian(mu, logvar)
            KL_loss = self.beta*total_kld
        elif self.z_dim_gauss == 0:
            total_kld, dim_wise_kld, mean_kld = kl_divergence_bernoulli(p_dist)
            KL_loss = self.gamma *total_kld 
        elif self.z_dim_bern !=0 and self.z_dim_gauss != 0:
            total_kld_bern, dim_wise_kld_bern, mean_kld_bern = kl_divergence_bernoulli(p_dist)
            total_kld_gauss, dim_wise_kld_gauss, mean_kld_gauss = kl_divergence_gaussian(mu, logvar)
            KL_loss = self.gamma *total_kld_bern + self.beta*total_kld_gauss
        return([recon_loss,KL_loss ])
    
    def test_loss(self):
        print("Calculating test loss")
        testLoss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        cnt = 0
    
        with torch.no_grad():
            for sample in self.test_dl:
                img = sample['x'].to(self.device)
                trgt = sample['y'].to(self.device)
                testLoss_list = self.run_model(img, trgt)
                recon_loss += testLoss_list[0]
                kl_loss += testLoss_list[1]
                cnt += 1
        testLoss += recon_loss + kl_loss
        testLoss = testLoss.div(cnt)
        self.testLoss = float(testLoss.cpu().numpy())
        recon_loss = recon_loss.div(cnt)
        self.test_recon_loss = float(recon_loss.cpu().numpy())
        kl_loss = kl_loss.div(cnt)
        self.test_kl_loss = float(kl_loss.cpu().numpy())
       
        print('[{}] test_Loss:{:.3f}, test_recon_loss:{:.3f}, test_KL_loss:{:.3f}'.format(
            self.global_iter, self.testLoss,self.test_recon_loss, self.test_kl_loss))
        return self.testLoss
    
    
    def gnrl_loss(self):
        print("Calculating generalisation loss")
        gnrlLoss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        cnt = 0
        with torch.no_grad():
            for sample in self.gnrl_dl:
                img = sample['x'].to(self.device)
                trgt = sample['y'].to(self.device)
                gnrlLoss_list = self.run_model(img, trgt)
                recon_loss += gnrlLoss_list[0]
                kl_loss += gnrlLoss_list[1]
                cnt += 1
        grnlLoss = recon_loss + kl_loss
        grnlLoss = grnlLoss.div(cnt)
        self.grnlLoss = float(grnlLoss.cpu().numpy())  #[0]
        recon_loss = recon_loss.div(cnt)
        self.gnrl_recon_loss = float(recon_loss.cpu().numpy()) #[0]
        kl_loss = kl_loss.div(cnt)
        self.gnrl_kl_loss = float(kl_loss.cpu().numpy()) #[0]
        
        print('[{}] gnrl_Loss:{:.3f} gnrl_recon_loss:{:.3f} gnrl_KL_loss:{:.3f}'.format(
            self.global_iter, self.grnlLoss, self.gnrl_recon_loss, self.gnrl_kl_loss))
        
    def test_plots(self):
        #self.net.eval()   but supposed to add when testing?
        net_copy = deepcopy(self.net)
        net_copy.to('cpu')
        
        print("creating sample images!")
        #Print sample images by decoding samples of normal distribution size of z_dim
        if self.z_dim_bern == 0:         
            sample = torch.randn(16, self.z_dim_gauss)
        elif self.z_dim_gauss == 0:         
            sample = torch.rand(16, self.z_dim_bern)
        elif self.z_dim_bern !=0 and self.z_dim_gauss != 0:
            sample_2 = torch.randn(16, self.z_dim_gauss)
            sample_1 = torch.rand(16, self.z_dim_bern)
            sample = torch.cat((sample_1,sample_2), 1)
        with torch.no_grad():
            if self.sbd == True:
                sbd_decoder = SB_decoder(self.z_dim_bern, self.z_dim_gauss, self.n_filter, self.nc)
                sbd_model = spatial_broadcast_decoder()
                sample = sbd_model(sample)
                test_recon = sbd_decoder(sample)
            else:
                print(sample)
                test_recon = net_copy._decode(sample)
        torchvision.utils.save_image( F.sigmoid(test_recon).view(
            test_recon.size(0),1, self.image_size, self.image_size).data.cpu(), '{}/sampling_z_{}.png'.
                                        format(self.output_dir, self.global_iter))        
        
        print("Constructing Z hist!")
        construct_z_hist(net_copy, self.test_dl, self.global_iter, self.output_dir,dim='depth')

        
        #select test image to traverse 
        print("Traversing!")
        with torch.no_grad():
            for i in range(3):
                example_id = self.test_data.__getitem__(i+random.randint(0,20))
                traverse_z(net_copy, example_id, ID=str(i),output_dir=self.output_dir, 
                           global_iter=self.global_iter, sbd = self.sbd, num_frames=100 )
    
        #create pdf with reconstructed test images 
        print('Reconstructing Test Images!')
        with torch.no_grad():
            plotsave_tests(net_copy, self.test_data, self.output_dir, self.global_iter,type="Test",  n=20 )
        
        print('Reconstructing generalisation images!')
        with torch.no_grad():
            plotsave_tests(net_copy, self.gnrl_data, self.output_dir, self.global_iter, type="Gnrl", n=20 )

    
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
            
        file_path_2 = os.path.join(self.output_dir, filename)
        if os.path.isfile(file_path_2):
            self.gather.load_data(file_path_2)
       
  

  