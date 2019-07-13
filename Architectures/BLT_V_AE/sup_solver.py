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


from dataset_sup import return_data_sup_encoder, return_data_sup_decoder
from model_BLT_VAE import BLT_mod, BLT_orig, FF

def supervised_encoder_loss(output, target, encoder_target_type):
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

def supervised_decoder_loss(img, recon):
    #reconstruction loss for GAUSSIAN distribution of pixel values (not Bernoulli)
    batch_size = recon.size(0)
    assert batch_size != 0
    recon = F.sigmoid(recon)
    recon_loss = F.mse_loss(recon, img, size_average=False).div(batch_size) #divide mse loss by batch size
    return recon_loss

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
        
    def save_data(self, glob_iter, output_dir, name ):
        if name == 'last':
            pickle.dump( self.data, open( "{}/data_{}.p".format(output_dir,name ), "wb" ) )
        else:
            pickle.dump( self.data, open( "{}/data_{}.p".format(output_dir, glob_iter ), "wb" ) )

    def load_data(self, glob_iter, output_dir, name):
        if name=='last':
            self.data = pickle.load( open( "{}/data_{}.p".format(output_dir,name ), "rb" ) )
        else:
            self.data = pickle.load( open( "{}/data_{}.p".format(output_dir,glob_iter ), "rb" ) )
            
            
class Solver_sup(object):
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
            
            
        self.n_filter = args.n_filter
        self.image_size = args.image_size
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.l2_loss= args.l2_loss
        
        if args.dataset.lower() == 'digits_gray':
            self.nc = 1
        elif args.dataset.lower() == 'digits_col':
            self.nc = 3
        else:
            raise NotImplementedError
            
        if args.model == 'FF':
            print("Using Feed forward model!")
            net = FF(z_dim=self.z_dim, nc=self.nc, n_filter=self.n_filter)
        elif args.model =='BLT_orig':
            print("Using original BLT model!")
            net = BLT_orig(z_dim=self.z_dim, nc=self.nc, batch_size=self.bs)
        elif args.model =='BLT_mod':
            print("Using modified BLT model!")
            net = BLT_mod(z_dim=self.z_dim, nc=self.nc, batch_size=self.bs)
        else:
            raise NotImplementedError('Model not correct')
        
        print("CUDA availability: " + str(torch.cuda.is_available()))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        # copy the model to each device
        self.net = net.to(self.device) 
        self.optim = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9) 
       
        
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
        self.output_dir = args.output_dir 
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
        #elif self.testing_method =='semisupervised':
        else:
            raise NotImplementedError    
        
    def train(self):
        #self.net(train=True)
        iters_per_epoch = len(self.train_dl)
        print(iters_per_epoch)
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
                #for i in range(x.size(0)):
                #    torchvision.utils.save_image( x[i,0,:,:] , '{}/x_{}_{}.png'.format(self.output_dir, self.global_iter, i)) 
                #    print(y[i,:])
                
                if self.testing_method =='supervised_encoder':
                    loss, final_out = self.run_model(self.testing_method, x, y, self.l2_loss)
                    train_accuracy = self.get_accuracy(final_out, y)
                elif self.testing_method == 'supervised_decoder':
                    loss, recon = self.run_model(self.testing_method, x, y, self.l2_loss)
                    
                self.adjust_learning_rate(self.optim, (count/iters_per_epoch))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                count +=1 
            
            
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.test_loss()
                    self.gather.insert(iter=self.global_iter, train_loss=loss.data, 
                                       test_loss = self.testLoss, test_accuracy = self.accuracy)
                
                if self.global_iter%self.display_step == 0:
                    if self.testing_method =='supervised_encoder': 
                        train_accuracy = self.get_accuracy(final_out, y)
                        print('[{}] train accuracy:{:.3f}'.format(self.global_iter, train_accuracy))
                    print('[{}] train loss:{:.3f}'.format(self.global_iter, torch.mean(loss)))
                        
                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    self.test_loss()
                    with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                        myfile.write('\n[{}] train_loss:{:.3f}, train_accuracy:{:.3f}, test_loss:{:.3f}, test_accuracy:{:.3f}'.format(
                            self.global_iter, torch.mean(loss), train_accuracy, self.testLoss, self.test_accuracy))
                  
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
        cnt = 0
        with torch.no_grad():
            for sample in self.test_dl:
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                
                testLoss_list= self.run_model(self.testing_method, x, y, self.l2_loss)
                if self.testing_method =='supervised_encoder':
                    final_out =testLoss_list[1]
                    test_accuracy += self.get_accuracy(final_out,y)
                testLoss += testLoss_list[0]
                cnt += 1

        testLoss = testLoss.div(cnt)
        self.testLoss = testLoss.cpu().numpy()#[0]
        self.test_accuracy = test_accuracy / cnt
        print('[{}] test_Loss:{:.3f}'.format(self.global_iter, self.testLoss))
        print('[{}] test accuracy:{:.3f}'.format(self.global_iter, self.test_accuracy))
    
    
    def get_accuracy(self, outputs, targets):
        assert outputs.size() == targets.size()
        assert outputs.size(0) > 0
        x = torch.topk(outputs,2,dim=1 )
        y = torch.topk(targets,2,dim=1 )
        outputs = x[1].cpu().numpy()
        targets  = y[1].cpu().numpy()

        accuracy = np.sum(outputs == targets)/outputs.size *100
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

            
