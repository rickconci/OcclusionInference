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
from data_loaders.dataset_sup import return_data_sup_encoder, return_data_sup_decoder
from models.BLT_models import multi_VAE, SB_decoder, spatial_broadcast_decoder

from solvers.visuals_mod import plot_decoder_img
from solvers.utils_mod import DataGather, get_accuracy
from solvers.losses import supervised_encoder_loss, supervised_decoder_loss
            
class Solver_sup(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        
        self.testing_method = args.testing_method
        self.encoder = args.encoder
        self.decoder = args.decoder
        self.n_filter = args.n_filter
        self.n_rep = args.n_rep
        self.kernel_size = args.kernel_size
        self.padding = args.padding
        self.sbd = args.sbd
        
        self.encoder_target_type = args.encoder_target_type
                
        if args.encoder_target_type == 'joint':
            self.z_dim = 10
        elif args.encoder_target_type == 'black_white':
            self.z_dim = 20
        elif args.encoder_target_type == 'depth_black_white':
            self.z_dim = 21
        elif args.encoder_target_type == 'depth_black_white_xy_xy':
            self.z_dim = 25
            
        
        if args.dataset.lower() == 'digits_gray':
            self.nc = 1
        elif args.dataset.lower() == 'digits_col':
            self.nc = 3
        else:
            raise NotImplementedError
        
        net = multi_VAE(self.encoder,self.decoder,self.z_dim, 0 ,self.n_filter,self.nc,
                        self.n_rep,self.sbd, self.kernel_size, self.padding)
        
        
        if self.sbd == True:
            self.decoder = SB_decoder(self.z_dim, 0, self.n_filter, self.nc)
            self.sbd_model = spatial_broadcast_decoder()
            
        
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
        if self.testing_method =='supervised_encoder':
            print(encoder_size ,"parameters in the encoder!")
            self.params = encoder_size
        elif self.testing_method =='supervised_decoder':
            print(decoder_size ,"parameters in the decoder!")
            self.params = decoder_size
        
        print("CUDA availability: " + str(torch.cuda.is_available()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        self.net = net.to(self.device) 
        print("net on cuda: " + str(next(self.net.parameters()).is_cuda))
        
        self.lr = args.lr
        self.l2_loss = args.l2_loss
        self.beta1 = args.beta1
        self.beta2 = args.beta2        
        if args.optim_type =='Adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif args.optim_type =='SGD':
            self.optim = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        
        if self.testing_method == 'supervised_encoder':
             self.train_dl, self.test_dl, self.gnrl_dl  = return_data_sup_encoder(args)
        elif self.testing_method == 'supervised_decoder':
            self.train_dl, self.test_dl, self.gnrl_dl , self.test_data, self.gnrl_data =  return_data_sup_decoder(args)
        else:
            raise NotImplementedError    
        
        self.max_epoch = args.max_epoch
        self.global_iter = 0
        self.max_epoch = args.max_epoch
        self.global_iter = 0
        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step
          
        self.image_size = args.image_size

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
        self.output_dir = args.output_dir 
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.ckpt_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        
        
        self.gather = DataGather(self.testing_method, self.encoder_target_type)
        
        
    def train(self):
        #self.net(train=True)
        iters_per_epoch = len(self.train_dl)
        print(iters_per_epoch, 'iters per epoch')
        max_iter = self.max_epoch*iters_per_epoch
        batch_size = self.train_dl.batch_size
        
        count = 0
        out = False
        pbar = tqdm(total=max_iter)
        pbar.update(self.global_iter)
        
        while not out:
            for sample in self.train_dl:
                self.global_iter += 1
                pbar.update(1)
            
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                
                #print(x.shape)
                #print(y.shape)
                #for i in range(x.size(0)):
                #    #print(x[i,:])
                #    torchvision.utils.save_image( x[i,0,:,:] , '{}/x_{}_{}.png'.format(self.output_dir, self.global_iter, i)) 
                 #   print(y[i,:])
                
                if self.testing_method =='supervised_encoder':
                    loss, final_out = self.run_model(self.testing_method, x, y, self.l2_loss)
                
                elif self.testing_method == 'supervised_decoder':
                    x = x.type(torch.FloatTensor).to(self.device)
                    loss, recon = self.run_model(self.testing_method, x, y, self.l2_loss)
                    
                self.adjust_learning_rate(self.optim, (count/iters_per_epoch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                count +=1 
            
            
                if self.global_iter%self.gather_step == 0:
                    self.test_loss()
                    self.gnrl_loss()
                  
                    if self.testing_method =='supervised_encoder': 
                        if self.encoder_target_type== 'joint':
                            self.gather.insert(iter=self.global_iter, train_loss=loss.data, 
                                           test_loss = self.testLoss,gnrl_loss = self.gnrlLoss,
                                               train_accuracy = train_accuracy, test_accuracy = self.test_accuracy,
                                              gnrl_accuracy = self.accuracy)
                        else:
                            accuracy_list = get_accuracy(final_out,y, self.encoder_target_type)
                            train_depth_accuracy = accuracy_list[0]
                            train_black_accuracy = accuracy_list[1]
                            train_white_accuracy = accuracy_list[2]
                            self.gather.insert(iter=self.global_iter, train_loss=loss.data, 
                                           test_loss = self.testLoss, gnrl_loss = self.gnrlLoss,
                                               train_depth_accuracy = train_depth_accuracy,
                                               train_black_accuracy = train_black_accuracy,
                                               train_white_accuracy= train_white_accuracy,
                                               test_depth_accuracy = self.test_depth_accuracy, 
                                               test_black_accuracy = self.test_black_accuracy,
                                               test_white_accuracy  = self.test_white_accuracy,
                                               gnrl_depth_accuracy = self.gnrl_depth_accuracy,
                                               gnrl_black_accuracy = self.gnrl_black_accuracy,
                                               gnrl_white_accuracy = self.gnrl_white_accuracy )
                        
                        
                        
                    elif self.testing_method =='supervised_decoder':
                        
                        self.gather.insert(iter=self.global_iter, train_recon_loss = torch.mean(loss), 
                                           test_recon_loss = self.testLoss, gnrl_recon_loss = self.gnrlLoss)
                        
                        with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                            myfile.write('\n[{}] train_recon_loss:{:.3f}, test_recon_loss:{:.3f} , gnrl_recon_loss:{:.3f}'.format(self.global_iter, torch.mean(loss), self.testLoss, self.gnrlLoss))
                
                
                if self.global_iter%self.display_step == 0:
                    print('[{}] train loss:{:.3f}'.format(self.global_iter, torch.mean(loss)))
                    if self.testing_method =='supervised_encoder': 
                        if self.encoder_target_type== 'joint':
                            train_accuracy = get_accuracy(final_out, y, self.encoder_target_type)
                            print('[{}] train accuracy:{:.3f}'.format(self.global_iter, train_accuracy))
                        else:
                            accuracy_list = get_accuracy(final_out,y,self.encoder_target_type)
                            train_depth_accuracy = accuracy_list[0]
                            train_black_accuracy = accuracy_list[1]
                            train_white_accuracy = accuracy_list[2]
                            
                            print('[{}], train_depth_accuracy:{:.3f}, train_black_accuracy:{:.3f}, train_white_accuracy:{:.3f}'.format(self.global_iter, train_depth_accuracy, train_black_accuracy, train_white_accuracy))
                            
                    
                        
                
                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last') 
                    
                    oldtestLoss = self.testLoss
                    self.test_loss()
                    print('old test loss', oldtestLoss,'current test loss', self.testLoss )
                    
                    if self.gnrl_dl != 0:
                        oldgnrlLoss = self.gnrlLoss
                        self.gnrl_loss()
                        print('old gnrl loss', oldgnrlLoss,'current gnrl loss', self.gnrlLoss )
                        if self.gnrlLoss < oldgnrlLoss:
                            self.save_checkpoint('best_gnrl')
                            pbar.write('Saved best GNRL checkpoint(iter:{})'.format(self.global_iter))
                        
                    if self.testLoss  < oldtestLoss or self.gnrlLoss < oldgnrlLoss:
                        self.save_checkpoint('best_test')
                        pbar.write('Saved best TEST checkpoint(iter:{})'.format(self.global_iter))
                    
                    if self.testing_method == 'supervised_decoder':
                        self.test_images()
                    
                    self.gather.save_data(self.global_iter, self.output_dir, 'last' )
                    
                    
                if self.global_iter%5000 == 0:
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
        #print(lr)
        
    def run_model(self, testing_method, x, y, l2_loss):
        if self.testing_method == 'supervised_encoder':
            final_out = self.net._encode(x)
            loss = supervised_encoder_loss(final_out, y, self.encoder_target_type)
            l2 = 0
            for p in self.net.parameters():
                l2 = l2 + p.pow(2).sum() #*0.5
            loss = loss + l2_loss * l2
            return([loss, final_out])
        
        elif self.testing_method =='supervised_decoder':
            if self.sbd:
                x = self.sbd_model(x)
                x = x.to(self.device)
            recon = self.net._decode(x)
            loss = supervised_decoder_loss(y, recon)
            l2 = 0
            for p in self.net.parameters():
                l2 = l2 + p.pow(2).sum() #*0.5
            loss = loss + l2_loss * l2
            return([loss, recon])
       
    def test_loss(self):
        print("Calculating test loss")
        testLoss = 0.0
        test_accuracy=0.0
        depth_accuracy = 0.0
        black_accuracy = 0.0
        white_accuracy = 0.0
        cnt = 0
        
        with torch.no_grad():
            for sample in self.test_dl:
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                
                if self.testing_method =='supervised_encoder':
                    testLoss_list= self.run_model(self.testing_method, x, y, self.l2_loss)
                    final_out =testLoss_list[1]
                    if self.encoder_target_type== 'joint':
                        test_accuracy += get_accuracy(final_out,y,self.encoder_target_type)
                    else:
                        accuracy_list = get_accuracy(final_out,y,self.encoder_target_type)
                        depth_accuracy += accuracy_list[0]
                        black_accuracy += accuracy_list[1]
                        white_accuracy += accuracy_list[2]
                elif self.testing_method =='supervised_decoder':
                    x = x.type(torch.FloatTensor).to(self.device)
                    testLoss_list= self.run_model(self.testing_method, x, y, self.l2_loss)
                    test_accuracy = 0
                testLoss += testLoss_list[0]
                cnt += 1

        testLoss = testLoss.div(cnt)
        self.testLoss = testLoss.cpu().numpy()#[0]
        print('[{}] test_Loss:{:.3f}'.format(self.global_iter, self.testLoss))

        if self.encoder_target_type== 'joint':
            self.test_accuracy = test_accuracy / cnt
            print('[{}] test accuracy:{:.3f}'.format(self.global_iter, self.test_accuracy))
        else:
            self.test_depth_accuracy = depth_accuracy/cnt
            self.test_black_accuracy = black_accuracy/cnt
            self.test_white_accuracy = white_accuracy/cnt
            print('[{}] test_depth_accuracy:{:.3f}, test_black_accuracy:{:.3f}, test_white_accuracy:{:.3f}'.format(self.global_iter, self.test_depth_accuracy, self.test_black_accuracy,self.test_white_accuracy))
    
       
   
    
    def gnrl_loss(self):
        print("Calculating generalisation loss")
        gnrlLoss = 0.0
        gnrl_accuracy = 0.0
        depth_accuracy = 0.0
        black_accuracy = 0.0
        white_accuracy = 0.0
        cnt = 0
        with torch.no_grad():
            for sample in self.gnrl_dl:
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                    
                if self.testing_method =='supervised_encoder':
                    grnlLoss_list= self.run_model(self.testing_method, x, y, self.l2_loss)
                    final_out =grnlLoss_list[1]
                    if self.encoder_target_type== 'joint':
                        test_accuracy += get_accuracy(final_out,y,self.encoder_target_type)
                    else:
                        accuracy_list = get_accuracy(final_out,y,self.encoder_target_type)
                        depth_accuracy += accuracy_list[0]
                        black_accuracy += accuracy_list[1]
                        white_accuracy += accuracy_list[2]
                elif self.testing_method =='supervised_decoder':
                    x = x.type(torch.FloatTensor).to(self.device)
                    grnlLoss_list= self.run_model(self.testing_method, x, y, self.l2_loss)
                    grnl_accuracy = 0
                gnrlLoss += grnlLoss_list[0]
                cnt += 1
                
        gnrlLoss = gnrlLoss.div(cnt)
        self.gnrlLoss = gnrlLoss.cpu().numpy()
        print('[{}] gnrl_Loss:{:.3f}'.format(self.global_iter, self.gnrlLoss))
        if self.encoder_target_type== 'joint':
            self.gnrl_accuracy = gnrl_accuracy / cnt
            print('[{}] gnrl accuracy:{:.3f}'.format(self.global_iter, self.gnrl_accuracy))
        else:
            self.gnrl_depth_accuracy = depth_accuracy/cnt
            self.gnrl_black_accuracy = black_accuracy/cnt
            self.gnrl_white_accuracy = white_accuracy/cnt
            print('[{}] gnrl_depth_accuracy:{:.3f}, gnrl_black_accuracy:{:.3f}, gnrl_white_accuracy:{:.3f}'.format(self.global_iter, self.gnrl_depth_accuracy, self.gnrl_black_accuracy,self.gnrl_white_accuracy))
            

    def test_images(self):
        net_copy = deepcopy(self.net)
        net_copy.to('cpu')
        
        print('Reconstructing test Images!')
        with torch.no_grad():
            plot_decoder_img(net_copy, self.test_data, self.output_dir, self.global_iter, self.sbd, type="test", n=20 )
            print('Reconstructing gnrl Images!')
            plot_decoder_img(net_copy, self.gnrl_data, self.output_dir, self.global_iter, self.sbd, type="gnrl", n=20 )

        
    
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

            
