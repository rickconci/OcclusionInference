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
import pandas as pd

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
#sys.path.insert(0, '/Users/riccardoconci/Desktop/code/ZuckermanProject/OcclusionInference/Architectures')
sys.path.insert(0, '/home/riccardo/Desktop/OcclusionInference/Architectures')
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
        self.AE = args.AE
        
        self.encoder_target_type = args.encoder_target_type
                
        data_csv = "{}digts.csv".format(args.dset_dir)
        data = pd.read_csv(data_csv, header=None)
        if  data.shape[1] > 37:
            self.n_digits =3
        elif data.shape[1] <= 37:
            self.n_digits = 2
            
        if args.encoder_target_type == 'joint':
            self.z_dim = 10
        elif args.encoder_target_type == 'black_white':
            self.z_dim = 20
        elif args.encoder_target_type == 'depth_black_white':
            self.z_dim = 21
        elif args.encoder_target_type == 'depth_black_white_xy_xy':
            self.z_dim = 25
        elif args.encoder_target_type== "depth_ordered_one_hot":
            if self.n_digits == 2:
                self.z_dim = 20
            elif self.n_digits ==3:
                self.z_dim = 30
        elif args.encoder_target_type== "depth_ordered_one_hot_xy":
            if self.n_digits == 2:
                self.z_dim = 24
            elif self.n_digits ==3:
                self.z_dim = 36
        
        if args.dataset.lower() == 'digits_gray':
            self.nc = 1
        elif args.dataset.lower() == 'digits_col':
            self.nc = 3
        else:
            raise NotImplementedError
        
        net = multi_VAE(self.encoder,self.decoder,self.z_dim, 0 ,self.n_filter,self.nc,
                        self.n_rep,self.sbd, self.kernel_size, self.padding, self.AE)
        
        
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
            print(encoder_size ,"parameters in the ENCODER!")
            self.params = encoder_size
        
        elif self.testing_method =='supervised_decoder':
            print(decoder_size ,"parameters in the DECODER!")
            self.params = decoder_size
            
        
        print("CUDA availability: " + str(torch.cuda.is_available()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        self.net = net.to(self.device) 
        print("net on cuda: " + str(next(self.net.parameters()).is_cuda))
        
           
        if self.testing_method == 'supervised_encoder':
            self.train_dl, self.gnrl_dl  = return_data_sup_encoder(args)
        elif self.testing_method == 'supervised_decoder':
            self.train_dl, self.gnrl_dl , self.gnrl_data =  return_data_sup_decoder(args)
        else:
            raise NotImplementedError    

          
        
        
        self.lr = args.lr
        self.l2_loss = args.l2_loss
        self.beta1 = args.beta1
        self.beta2 = args.beta2        
        if args.optim_type =='Adam':
            self.optim = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        elif args.optim_type =='SGD':
            self.optim = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)    
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
        
        
        self.gather = DataGather(self.testing_method, self.encoder_target_type, self.n_digits)
        
        
    def train(self):
        #self.net(train=True)
        iters_per_epoch = len(self.train_dl)
        print(iters_per_epoch, 'iters per epoch')
        max_iter = self.max_epoch*iters_per_epoch
        batch_size = self.train_dl.batch_size
        oldgnrlLoss = Inf
        
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
                #    print(x[i,:])
                #    torchvision.utils.save_image( y[i,0,:,:] , '{}/x_{}_{}.png'.format(self.output_dir, self.global_iter, i)) 
                #   print(y[i,:])
                
                if self.testing_method =='supervised_encoder':
                    loss, final_loss_list, final_out, train_l2_reg_loss = self.run_model(self.testing_method, x, y, self.l2_loss)
                elif self.testing_method == 'supervised_decoder':
                    x = x.type(torch.FloatTensor).to(self.device)
                    loss, loss_list, recon, train_l2_reg_loss = self.run_model(self.testing_method, x, y, self.l2_loss)
                    loss_list = [round(x.item(),3) for x in loss_list]
                
                self.adjust_learning_rate(self.optim, (count/iters_per_epoch))
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                count +=1 
                
            
                if self.global_iter%self.gather_step == 0:
                    gnrlLoss = self.gnrl_loss()
                  
                    if self.testing_method =='supervised_encoder': 
                        
                        if self.encoder_target_type== 'joint':
                            self.gather.insert(iter=self.global_iter, train_loss=loss.data.item(), 
                                               gnrl_loss = gnrlLoss,
                                               train_accuracy = train_accuracy,
                                                gnrl_accuracy = self.accuracy)
                        elif self.encoder_target_type== "depth_black_white" or self.encoder_target_type== "depth_black_white_xy_xy":
                            accuracy_list = get_accuracy(final_out,y, self.encoder_target_type , self.n_digits)
                            train_depth_accuracy = accuracy_list[0]
                            train_black_accuracy = accuracy_list[1]
                            train_white_accuracy = accuracy_list[2]
                            self.gather.insert(iter=self.global_iter, train_loss=loss.data.item(), 
                                               gnrl_loss = gnrlLoss,
                                               train_depth_accuracy = train_depth_accuracy,
                                               train_black_accuracy = train_black_accuracy,
                                               train_white_accuracy= train_white_accuracy,
                                               gnrl_depth_accuracy = self.gnrl_depth_accuracy,
                                               gnrl_black_accuracy = self.gnrl_black_accuracy,
                                               gnrl_white_accuracy = self.gnrl_white_accuracy,
                                              depth_loss = float(final_loss_list[1]),
                                              black_loss = float(final_loss_list[2]),
                                              white_loss = float(final_loss_list[3]),
                                              xy_loss = float(final_loss_list[4]))
                        elif self.encoder_target_type== "depth_ordered_one_hot" or self.encoder_target_type== "depth_ordered_one_hot_xy" :
                            if self.n_digits ==2:
                                accuracy_list = get_accuracy(final_out,y, self.encoder_target_type, self.n_digits)
                                train_back_accuracy = accuracy_list[0]
                                train_front_accuracy = accuracy_list[1]
                                self.gather.insert(iter=self.global_iter, train_loss=loss.data.item(), 
                                                   gnrl_loss = gnrlLoss,
                                                   l2_reg_loss = train_l2_reg_loss.item(),
                                                   
                                                   train_back_accuracy = train_back_accuracy,
                                                   train_front_accuracy= train_front_accuracy,
                                                   gnrl_back_accuracy = self.gnrl_back_accuracy,
                                                   gnrl_front_accuracy = self.gnrl_front_accuracy,
                                                   
                                                   train_tot_final_iter_loss = float(final_loss_list[0]) + train_l2_reg_loss,
                                                   train_back_loss = float(final_loss_list[1]),
                                                   train_front_loss = float(final_loss_list[3]),
                                                   train_xy_loss = float(final_loss_list[4]),
                                                   
                                                   gnrl_tot_final_iter_loss = self.gnrl_total_last_iter_loss,
                                                   gnrl_back_loss = self.gnrl_back_loss,
                                                   gnrl_front_loss = self.gnrl_front_loss,
                                                   gnrl_xy_loss= self.gnrl_xy_loss
                                                  )
                                
                            elif self.n_digits ==3:
                                accuracy_list = get_accuracy(final_out,y, self.encoder_target_type, self.n_digits)
                                train_back_accuracy = accuracy_list[0]
                                train_mid_accuracy = accuracy_list[1]
                                train_front_accuracy = accuracy_list[2]
                                self.gather.insert(iter=self.global_iter, train_loss=loss.data.item(), 
                                                   gnrl_loss = gnrlLoss,
                                                   l2_reg_loss = train_l2_reg_loss,
                                                   
                                                   train_back_accuracy = train_back_accuracy,
                                                   train_mid_accuracy = train_mid_accuracy,
                                                   train_front_accuracy= train_front_accuracy,
                                                   gnrl_back_accuracy = self.gnrl_back_accuracy,
                                                   gnrl_mid_accuracy = self.gnrl_mid_accuracy,
                                                   gnrl_front_accuracy = self.gnrl_front_accuracy,
                                                   
                                                   train_tot_final_iter_loss = float(final_loss_list[0]) + train_l2_reg_loss,
                                                   train_back_loss = float(final_loss_list[1]),
                                                   train_mid_loss = float(final_loss_list[2]),
                                                   train_front_loss = float(final_loss_list[3]),
                                                   train_xy_loss = float(final_loss_list[4]),
                                                   
                                                   gnrl_tot_final_iter_loss = self.gnrl_total_last_iter_loss,
                                                   gnrl_back_loss = self.gnrl_back_loss,
                                                   gnrl_mid_loss = self.gnrl_mid_loss,
                                                   gnrl_front_loss = self.gnrl_front_loss,
                                                   gnrl_xy_loss=  self.gnrl_xy_loss,
                                                   
                                                  )
                                with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                                    myfile.write('\n[{}] train_loss:{:.3f}, gnrl_loss:{:.3f}'.format(self.global_iter,loss.data.item(), gnrlLoss))
                            
                        
                        
                    elif self.testing_method =='supervised_decoder':
                        if self.decoder =='B':
                            self.gather.insert(iter=self.global_iter, train_recon_loss = loss.item(), gnrl_recon_loss = gnrlLoss)
                            with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                                myfile.write('\n[{}] train_recon_loss:{:.3f}, gnrl_recon_loss:{:.3f}'.format(self.global_iter, loss.item(), gnrlLoss))
                        else:
                            self.gather.insert(iter=self.global_iter, train_recon_loss = loss.item(), gnrl_recon_loss = gnrlLoss, train_recon_last_iter_loss=loss_list[-2], gnrl_total_last_iter_loss= self.gnrl_total_last_iter_loss)
                            with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                                myfile.write('\n[{}] train_recon_loss:{:.3f}, gnrl_recon_loss:{:.3f}, {}'.format(self.global_iter, torch.mean(loss), gnrlLoss,loss_list))
                
                
                if self.global_iter%self.display_step == 0:
                    print('[{}] train loss:{:.3f}'.format(self.global_iter, loss.item()))
                    
                    if self.testing_method =='supervised_encoder': 
                        if self.encoder_target_type== 'joint':
                            train_accuracy = get_accuracy(final_out, y, self.encoder_target_type, self.n_digits)
                            print('[{}] train accuracy:{:.3f}'.format(self.global_iter, train_accuracy))
                        elif self.encoder_target_type== "depth_black_white" or self.encoder_target_type== "depth_black_white_xy_xy":
                            accuracy_list = get_accuracy(final_out,y,self.encoder_target_type, self.n_digits)
                            train_depth_accuracy = accuracy_list[0]
                            train_black_accuracy = accuracy_list[1]
                            train_white_accuracy = accuracy_list[2]
                            
                            print('[{}], train_depth_accuracy:{:.3f}, train_black_accuracy:{:.3f}, train_white_accuracy:{:.3f}'.format(self.global_iter, train_depth_accuracy, train_black_accuracy, train_white_accuracy))
                        elif self.encoder_target_type== "depth_ordered_one_hot" or self.encoder_target_type== "depth_ordered_one_hot_xy" :
                            if self.n_digits ==2:
                                accuracy_list = get_accuracy(final_out,y, self.encoder_target_type, self.n_digits)
                                train_back_accuracy = accuracy_list[0]
                                train_front_accuracy = accuracy_list[1]
                                print('[{}], train_back_accuracy:{:.3f}, train_front_accuracy:{:.3f}'.format(self.global_iter, train_back_accuracy, train_front_accuracy))
                                
                            elif self.n_digits ==3:
                                accuracy_list = get_accuracy(final_out,y, self.encoder_target_type , self.n_digits)
                                train_back_accuracy = accuracy_list[0]
                                train_mid_accuracy = accuracy_list[1]
                                train_front_accuracy = accuracy_list[2]

                                print('[{}], train_back_accuracy:{:.3f}, train_mid_accuracy:{:.3f}, train_front_accuracy:{:.3f}'.format(self.global_iter, train_back_accuracy, train_mid_accuracy, train_front_accuracy))
                    elif self.testing_method =='supervised_decoder':
                        if self.decoder != 'B':
                            print([round(x,3) for x in loss_list])

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last') 
                    
                    if self.gnrl_dl != 0:
                        gnrlLoss = self.gnrl_loss()
                        print('old gnrl loss', oldgnrlLoss,'current gnrl loss', gnrlLoss )
                        if gnrlLoss < oldgnrlLoss:
                            oldgnrlLoss = gnrlLoss
                            self.save_checkpoint('best_gnrl')
                            pbar.write('Saved best GNRL checkpoint(iter:{})'.format(self.global_iter))
                    
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
        """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
        lr = self.lr * (0.1 ** (epoch / 40))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        #print(lr)
        
    def run_model(self, testing_method, x, y, l2_coeff):
        if self.testing_method == 'supervised_encoder':
            final_out_list = self.net._encode(x)
            loss = 0.0
            for i in range(len(final_out_list)):
                final_loss_list = supervised_encoder_loss(final_out_list[i], y, self.n_digits, self.encoder_target_type)
                #if self.global_iter%self.display_step == 0:
                #    print('iter {}'.format(i), [x.item() for x in final_loss_list])
                loss = final_loss_list[0]  #no more composite loss
                
            l2 = 0
            for p in self.net.parameters():
                l2 = l2 + p.pow(2).sum() #*0.5
            loss = loss + l2_coeff * l2
            return([loss, final_loss_list, final_out_list[-1], l2_coeff*l2])
        
        elif self.testing_method =='supervised_decoder':
            if self.sbd:
                x = self.sbd_model(x)
                x = x.to(self.device)
            recon_list = self.net._decode(x)
            loss_list = []
            loss = 0.0
            for i in range(len(recon_list)):
                loss_list.append(supervised_decoder_loss(y, recon_list[i]))
                loss = supervised_decoder_loss(y, recon_list[i]) #no more composite loss
            l2 = 0
            for p in self.net.parameters():
                l2 = l2 + p.pow(2).sum() #*0.5
            loss = loss + l2_coeff * l2
            return([ loss, loss_list, recon_list[-1], l2_coeff*l2])

    def gnrl_loss(self):
        print("Calculating generalisation loss")
        gnrlLoss = 0.0
        gnrl_accuracy = 0.0
        depth_accuracy = 0.0
        black_accuracy = 0.0
        white_accuracy = 0.0
        gnrl_back_accuracy = 0.0
        gnrl_mid_accuracy = 0.0
        gnrl_front_accuracy =0.0
        gnrl_total_last_iter_loss= 0.0
        gnrl_back_loss = 0.0
        gnrl_front_loss = 0.0
        gnrl_xy_loss = 0.0
        gnrl_mid_loss = 0.0
        gnrl_l2_loss = 0.0
        gnrlLoss_set = []
        cnt = 0
        with torch.no_grad():
            for sample in self.gnrl_dl:
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                    
                if self.testing_method =='supervised_encoder':
                    grnlLoss_list = self.run_model(self.testing_method, x, y, self.l2_loss)
                    final_encoder_losses = grnlLoss_list[1]
                    final_encoder_losses = [x.item() for x in final_encoder_losses]
                    gnrl_back_loss += final_encoder_losses[1] 
                    gnrl_front_loss += final_encoder_losses[3]
                    gnrl_xy_loss += final_encoder_losses[4]
                    gnrl_l2_loss +=  grnlLoss_list[3]
                    
                    gnrl_total_last_iter_loss += final_encoder_losses[0]
                    gnrl_total_last_iter_loss += grnlLoss_list[3]
                    
 
                    if self.n_digits ==3:
                        gnrl_mid_loss += final_encoder_losses[2]
                    
                    final_out = grnlLoss_list[2]
                    
                    if self.encoder_target_type== 'joint':
                        test_accuracy += get_accuracy(final_out,y,self.encoder_target_type, self.n_digits)
                    elif self.encoder_target_type== "depth_black_white" or self.encoder_target_type== "depth_black_white_xy_xy":
                        accuracy_list = get_accuracy(final_out,y,self.encoder_target_type, self.n_digits)
                        depth_accuracy += accuracy_list[0]
                        black_accuracy += accuracy_list[1]
                        white_accuracy += accuracy_list[2]
                    elif self.encoder_target_type== "depth_ordered_one_hot" or self.encoder_target_type== "depth_ordered_one_hot_xy" :
                        if self.n_digits ==2:
                            accuracy_list = get_accuracy(final_out,y, self.encoder_target_type, self.n_digits)
                            gnrl_back_accuracy += accuracy_list[0]
                            gnrl_front_accuracy += accuracy_list[1]
                                
                        elif self.n_digits ==3:
                            accuracy_list = get_accuracy(final_out,y, self.encoder_target_type, self.n_digits)
                            gnrl_back_accuracy += accuracy_list[0]
                            gnrl_mid_accuracy += accuracy_list[1]
                            gnrl_front_accuracy += accuracy_list[2]
                        
                elif self.testing_method =='supervised_decoder':
                    x = x.type(torch.FloatTensor).to(self.device)
                    grnlLoss_list = self.run_model(self.testing_method, x, y, self.l2_loss)
                    gnrlLoss_list_sep = grnlLoss_list[1]
                    gnrlLoss_list_sep = [x.item() for x in gnrlLoss_list_sep]
                    gnrl_total_last_iter_loss += gnrlLoss_list_sep[-1]
                    gnrl_total_last_iter_loss = gnrl_total_last_iter_loss + grnlLoss_list[-1]
                    gnrl_l2_loss +=  grnlLoss_list[-1]
                    grnl_accuracy = 0
                    #gnrlLoss_set = [sum(x) for x in zip(grnlLoss_list[1], gnrlLoss_set)]
                
                gnrlLoss += grnlLoss_list[0]

                cnt += 1
                
        gnrlLoss = gnrlLoss.div(cnt)
        gnrlLoss = gnrlLoss.cpu().item()
        self.gnrl_l2_loss = gnrl_l2_loss/cnt
        print('[{}] all iters gnrl_Loss:{:.3f}, l2_loss{:.3f}'.format(self.global_iter, gnrlLoss, self.gnrl_l2_loss))
        
        if self.testing_method =='supervised_decoder':
            self.gnrl_total_last_iter_loss = gnrl_total_last_iter_loss/cnt
            gnrlLoss_set = [x/cnt for x in gnrlLoss_set]
            print(gnrlLoss_set)
        elif self.testing_method =='supervised_encoder':
            if self.encoder_target_type== 'joint':
                self.gnrl_accuracy = gnrl_accuracy / cnt
                print('[{}] gnrl accuracy:{:.3f}'.format(self.global_iter, self.gnrl_accuracy))
            elif self.encoder_target_type== "depth_black_white" or self.encoder_target_type== "depth_black_white_xy_xy":
                self.gnrl_depth_accuracy = depth_accuracy/cnt
                self.gnrl_black_accuracy = black_accuracy/cnt
                self.gnrl_white_accuracy = white_accuracy/cnt
                print('[{}] gnrl_depth_accuracy:{:.3f}, gnrl_black_accuracy:{:.3f}, gnrl_white_accuracy:{:.3f}'.format(self.global_iter, self.gnrl_depth_accuracy, self.gnrl_black_accuracy,self.gnrl_white_accuracy))
            elif self.encoder_target_type== "depth_ordered_one_hot" or self.encoder_target_type== "depth_ordered_one_hot_xy" :
                self.gnrl_back_loss = gnrl_back_loss/cnt
                self.gnrl_front_loss = gnrl_front_loss/cnt
                self.gnrl_xy_loss = gnrl_xy_loss/cnt
                
                self.gnrl_total_last_iter_loss = gnrl_total_last_iter_loss/cnt
               

                if self.n_digits ==2:
                    self.gnrl_back_accuracy = gnrl_back_accuracy/cnt
                    self.gnrl_front_accuracy = gnrl_front_accuracy/cnt
                    print('[{}] tot_last_iter_gnrl_loss{:.3f}, last_iter_gnrl_back_loss:{:.3f}, last_iter_gnrl_front_loss:{:.3f}, last_iter_gnrl_xy_loss{:.3f}'.format(
                                                                                                                        self.global_iter, self.gnrl_total_last_iter_loss, 
                                                                                                                        self.gnrl_back_loss,self.gnrl_front_loss, self.gnrl_xy_loss))
                    print('[{}] gnrl_back_accuracy:{:.3f}, gnrl_front_accuracy:{:.3f}'.format(self.global_iter, self.gnrl_back_accuracy,self.gnrl_front_accuracy))
                elif self.n_digits ==3:
                    self.gnrl_mid_loss = gnrl_mid_loss/cnt
                    self.gnrl_back_accuracy = gnrl_back_accuracy/cnt
                    self.gnrl_mid_accuracy = gnrl_mid_accuracy/cnt
                    self.gnrl_front_accuracy = gnrl_front_accuracy/cnt
                    print('[{}] tot_last_iter_gnrl_loss{:.3f}, last_iter_gnrl_back_loss:{:.3f},last_iter_gnrl_mid_loss:{:.3f}, last_iter_gnrl_front_loss:{:.3f}, last_iter_gnrl_xy_loss{:.3f}'.format(
                                                                                                                        self.global_iter, self.gnrl_total_last_iter_loss, 
                                                                                                                        self.gnrl_back_loss,self.gnrl_mid_loss, self.gnrl_front_loss, self.gnrl_xy_loss))
                    print('[{}] gnrl_back_accuracy:{:.3f}, gnrl_mid_accuracy:{:.3f}, gnrl_front_accuracy:{:.3f}'.format(self.global_iter, self.gnrl_back_accuracy, self.gnrl_mid_accuracy,self.gnrl_front_accuracy))

        return(gnrlLoss)

    def test_images(self):
        net_copy = deepcopy(self.net)
        net_copy.to('cpu')
        
        with torch.no_grad():
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

            
