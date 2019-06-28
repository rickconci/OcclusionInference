"""solver_mod.py"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Image

import visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision.utils import make_grid, save_image


from dataset_mod import return_data, MyDataset
from model_mod import conv_VAE_32, conv_AE
from visuals_mod import traverse_z, plotsave_tests


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

def flatten(x):
    return to_var(x.view(x.size(0), -1))

def save_image(x, path='real_image.png'):
    torchvision.utils.save_image(x, path)


def reconstruction_loss(x, x_recon):
    #reconstruction loss for GAUSSIAN distribution of pixel values (not Bernoulli)
    batch_size = x.size(0)
    assert batch_size != 0
    x_recon = F.sigmoid(x_recon)
    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size) #divide mse loss by batch size
    return recon_loss


def kl_divergence(mu, logvar):
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

        
class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        
        self.model = args.model
        self.max_iter = args.max_iter
        self.global_iter = 0
        
        self.z_dim = args.z_dim
        self.n_filter = args.n_filter

        self.image_size = args.image_size
        self.beta = args.beta
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        
        if args.dataset.lower() == 'digits':
            self.nc = 1
        else:
            raise NotImplementedError
            
        if args.model == 'conv_VAE_32':
            net = conv_VAE_32(z_dim=self.z_dim,n_filter=self.n_filter, nc=self.nc, train=True)
        elif args.model == 'conv_AE':
            net = conv_AE(z_dim=self.z_dim,n_filter=self.n_filter, nc=self.nc, train=True)
            self.beta = 0
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
        self.train_data_loader, self.test_data_loader = return_data(args)

        self.gather = DataGather()
        
    def train(self):
        #self.net(train=True)
        if torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
            
        # copy the model to each device
        #self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.net = net.to(self.device) 

        out = False
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        
        while not out:
            for sample in self.train_data_loader: 
                self.global_iter += 1
                pbar.update(1)
                
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                
                if self.model == 'conv_VAE_32':
                    x_recon, mu, logvar = self.net(x)
                    recon_loss = reconstruction_loss(y, x_recon)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.model == 'conv_AE':
                    x_recon = self.net(x)
                    recon_loss = reconstruction_loss(y, x_recon)
                    beta_vae_loss = recon_loss
                                
                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()
                
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    if self.model == 'conv_VAE_32':
                        self.gather.insert(iter=self.global_iter,
                                           mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                           recon_loss=recon_loss.data, total_kld=total_kld.data,
                                           dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)
                    
                if self.global_iter%self.display_step == 0:
                    if self.model == 'conv_VAE_32':
                        pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                            self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0]))

                        var = logvar.exp().mean(0).data
                        var_str = ''
                        for j, var_j in enumerate(var):
                            var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                        pbar.write(var_str)
                    elif self.model == 'conv_AE':
                        pbar.write('[{}] recon_loss:{:.3f}'.format(self.global_iter, recon_loss.data))
                    
                    #if self.viz_on:
                    #    print("now visdoming!")
                    #    self.gather.insert(images=y.data)
                    #    self.gather.insert(images=F.sigmoid(x_recon).data)
                    #    self.viz_reconstruction()
                    #    self.viz_lines()
                    #    self.gather.flush()
                        
                if self.global_iter%self.save_step == 0:
                    self.test_loss()
                    #if self.testLoss  < 
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    self.test_plots()
                    if self.model == 'conv_VAE_32':
                        with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                            myfile.write('\n[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}, test_loss:{;.3f}'.format(self.global_iter, recon_loss.data, total_kld.data[0],mean_kld.data[0], self.testLoss))
                    elif self.model == 'conv_AE':
                        with open("{}/LOGBOOK.txt".format(self.output_dir), "a") as myfile:
                            myfile.write('\n[{}] recon_loss:{:.3f} test_loss:{;.3f}'.format(self.global_iter,
                                                                                            recon_loss.data,
                                                                                            self.testLoss))
                                     
                if self.global_iter%500 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    
    def test_loss(self):
        print("Calculating test loss")
        testLoss = 0.0
        cnt = 0
        with torch.no_grad():
            for sample in self.test_data_loader:
                x = sample['x'].to(self.device)
                y = sample['y'].to(self.device)
                print(x.shape)
                
                if self.model == 'conv_VAE_32':
                    x_recon, mu, logvar = self.net(x)
                    recon_loss = reconstruction_loss(y, x_recon)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                    testLoss += recon_loss + self.beta*total_kld
                elif self.model == 'conv_AE':
                    x_recon = self.net(x)
                    recon_loss = reconstruction_loss(y, x_recon)
                    testLoss += recon_loss
                print(cnt)
                cnt += 1
        self.testLoss = testLoss/cnt
        print('[{}] test_Loss:{:.3f}'.format(
                        self.global_iter, testLoss))
        
        #self.gather.insert(iter=self.global_iter,
        #                   mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
        #                   recon_loss=recon_loss.data, total_kld=total_kld.data,
        #                   dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data,
        #                   test_loss=testLoss)
        
    def test_plots(self):
        #self.net.eval()   but supposed to add when testing?
        self.net.to('cpu')

        #Print sample images by decoding samples of normal distribution size of z_dim
        sample = torch.randn(16, self.z_dim)
        with torch.no_grad():
            test_recon = self.net._decode(sample)
            torchvision.utils.save_image( F.sigmoid(test_recon).view(
                test_recon.size(0),1, self.image_size, self.image_size).data.cpu(), 'sample_image.png')
            Image('{}/sampling_z_{}.png'.format(self.output_dir, self.global_iter))
        
        
        #select test image to traverse 
        print("Traversing!")
        self.test_image_paths = os.path.join(self.dset_dir + "test/orig/")
        self.test_target_paths = os.path.join(self.dset_dir + "test/inverse/")
        dset = MyDataset
        test_data = dset(self.test_image_paths,self.test_target_paths, image_size= self.image_size)
        for i in range(3):
            example_id = test_data.__getitem__(i)
            traverse_z(self.net, example_id, ID=str(i),output_dir=self.output_dir, 
                       global_iter=self.global_iter, model= self.model, num_frames=200 )
    
        #create pdf with reconstructed test images 
        print('Reconstructing Test Images!')
        plotsave_tests(self.net, test_data, self.output_dir, self.global_iter, n=20 )

        
    #def net_mode(self, train):
    #    if not isinstance(train, bool):
    #        raise('Only bool type is supported. True or False')#
    #
    #    if train:
    #        self.net.train()
    #    else:
    #        self.net.eval()

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
