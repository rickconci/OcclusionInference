"""utils_mod.py"""

import argparse
import subprocess

import os
import numpy as np
import pickle

from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn


import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
    
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

    
def get_accuracy(outputs, targets, encoder_target_type, n_digits):
    assert outputs.size() == targets.size()
    assert outputs.size(0) > 0
    batch_size = outputs.size(0)
    if encoder_target_type== 'joint':
        x = torch.topk(outputs,2,dim=1 )[1]
        y = torch.topk(targets,2,dim=1 )[1]
        outputs = x[1]
        targets  = y[1]
        accuracy = torch.sum(outputs == targets)/outputs.size *100
        return(accuracy)
    elif encoder_target_type== "depth_black_white" or encoder_target_type== "depth_black_white_xy_xy":
        depth = F.sigmoid(outputs[:,0]).detach().round()
        #depth[depth<0.2] = 0
        #depth[depth>0.8] = 1
        depth_accuracy = torch.sum(depth == targets[:,0]).float()/batch_size *100
        black = torch.topk(outputs[:,1:11],1,dim=1 )[1]
        black_targets = torch.topk(targets[:,1:11],1,dim=1 )[1]
        black_accuracy = torch.sum(black == black_targets).float()/batch_size *100
        white = torch.topk(outputs[:,11:21],1,dim=1 )[1]
        white_targets = torch.topk(targets[:,11:21],1,dim=1 )[1]
        white_accuracy = torch.sum(white == white_targets).float()/batch_size *100
        depth_accuracy = depth_accuracy.cpu().numpy()
        black_accuracy = black_accuracy.cpu().numpy()
        white_accuracy = white_accuracy.cpu().numpy()
        return [depth_accuracy, black_accuracy, white_accuracy]
    elif encoder_target_type== "depth_ordered_one_hot" or encoder_target_type== "depth_ordered_one_hot_xy":
        if n_digits ==2:
            back_out = torch.topk(outputs[:,0:10], 1, dim=1)[1]
            back_target = torch.topk(targets[:,0:10],1,dim=1)[1]
            back_accuracy = torch.sum(back_out==back_target).float()/batch_size * 100

            front_out = torch.topk(outputs[:,10:20], 1, dim=1)[1]
            front_target = torch.topk(targets[:,10:20],1,dim=1)[1]
            front_accuracy = torch.sum(front_out == front_target).float()/batch_size * 100
            
            return([back_accuracy, front_accuracy])
        
        if n_digits ==3:
            
            m = nn.Softmax()
            
            back_out = torch.topk(m(outputs[:,0:10]), 1, dim=1)[1]
            back_target = torch.topk(targets[:,0:10],1,dim=1)[1]
            back_accuracy = torch.sum(back_out==back_target).float()/batch_size * 100
            
            mid_out = torch.topk(m(outputs[:,10:20]), 1, dim=1)[1]
            mid_target = torch.topk(targets[:,10:20],1,dim=1)[1]
            mid_accuracy= torch.sum(mid_out == mid_target).float()/batch_size*100
            
            front_out = torch.topk(m(outputs[:,20:30]), 1, dim=1)[1]
            front_target = torch.topk(targets[:,20:30],1,dim=1)[1]
            front_accuracy = torch.sum(front_out == front_target).float()/batch_size * 100
            
    
            return([back_accuracy, mid_accuracy, front_accuracy])
    
    

class DataGather(object):
    def __init__(self, testing_method, encoder_target_type, n_digts):
        self.encoder_target_type = encoder_target_type
        self.testing_method = testing_method
        self.n_digts = n_digts
        self.data = self.get_empty_data_dict()
        
    def get_empty_data_dict(self):
        if self.testing_method == 'unsupervised':
            return dict(iter=[],
                    trainLoss = [],
                    train_recon_loss=[],
                    train_KL_loss=[],
                    gnrlLoss=[],
                    gnrl_recon_loss=[],
                   gnrl_kl_loss=[])

        
        elif self.testing_method == 'supervised_encoder':
            if self.encoder_target_type== 'joint':
                return dict(iter=[],
                            train_loss = [],
                            gnrl_loss = [],
                            train_accuracy = [],
                            gnrl_accuracy = []
                           )
            elif self.encoder_target_type== "depth_black_white" or self.encoder_target_type== "depth_black_white_xy_xy":
                return dict(iter=[],
                            train_loss = [],
                            gnrl_loss = [],
                            train_depth_accuracy = [],
                            train_black_accuracy = [],
                            train_white_accuracy = [],
                            gnrl_depth_accuracy = [],
                            gnrl_black_accuracy = [],
                            gnrl_white_accuracy = [],
                            train_depth_loss = [],
                            train_black_loss = [],
                            train_white_loss = [],
                            train_xy_loss = []
                           )
            elif self.encoder_target_type== "depth_ordered_one_hot" or self.encoder_target_type== "depth_ordered_one_hot_xy" :
                if self.n_digts ==2:
                    return dict(iter=[],
                            train_loss = [],
                            gnrl_loss = [],
                            l2_reg_loss = [],
                                
                            train_back_accuracy = [],
                            train_front_accuracy = [],
                            gnrl_back_accuracy = [],
                            gnrl_front_accuracy = [],
                                
                            train_tot_final_iter_loss = [],
                            train_back_loss = [],
                            train_front_loss = [],
                            train_xy_loss = [],
                                
                            gnrl_tot_final_iter_loss =[],
                            gnrl_back_loss = [],
                            gnrl_front_loss = [],
                            gnrl_xy_loss = []
                           )
                
                    
                elif self.n_digts ==3:  
                    return dict(iter=[],
                            train_loss = [],
                            gnrl_loss = [],
                            l2_reg_loss = [],
                                
                            train_back_accuracy = [],
                            train_mid_accuracy = [],
                            train_front_accuracy = [],
                            gnrl_back_accuracy = [],
                            gnrl_mid_accuracy = [],
                            gnrl_front_accuracy = [],
                                
                            train_tot_final_iter_loss = [],
                            train_back_loss = [],
                            train_mid_loss = [],
                            train_front_loss = [],
                            train_xy_loss = [],
                                
                            gnrl_tot_final_iter_loss =[], 
                            gnrl_back_loss = [],
                            gnrl_mid_loss = [],
                            gnrl_front_loss = [],
                            gnrl_xy_loss = []
                           )
                
        elif self.testing_method =='supervised_decoder':
            return dict(iter=[],
                        train_recon_loss=[],
                        gnrl_recon_loss =[],
                        train_recon_last_iter_loss = [],
                        gnrl_total_last_iter_loss = []
                       )

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
            