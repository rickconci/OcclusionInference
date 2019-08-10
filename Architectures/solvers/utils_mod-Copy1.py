"""utils_mod.py"""

import argparse
import subprocess

import os
import numpy as np
import pickle

from torchvision import transforms
import torch
import torch.nn.functional as F

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

    
def get_accuracy(outputs, targets, encoder_target_type):
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
    else:
        depth = F.sigmoid(outputs[:,0]).detach()#.round()
        depth[depth<0.2] = 0
        depth[depth>0.8] = 1
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
    
    

class DataGather(object):
    def __init__(self, testing_method, encoder_target_type):
        self.encoder_target_type = encoder_target_type
        self.testing_method = testing_method
        self.data = self.get_empty_data_dict()
        

    def get_empty_data_dict(self):
        if self.testing_method == 'unsupervised':
            return dict(iter=[],
                    trainLoss = [],
                    train_recon_loss=[],
                    train_KL_loss=[],
                    testLoss=[],
                    test_recon_loss=[],
                    test_kl_loss=[],
                    gnrlLoss=[],
                    gnrl_recon_loss=[],
                   gnrl_kl_loss=[])

        
        elif self.testing_method == 'supervised_encoder':
            if self.encoder_target_type== 'joint':
                return dict(iter=[],
                            train_loss = [],
                            test_loss = [],
                            gnrl_loss = [],
                            train_accuracy = [],
                            test_accuracy = [],
                            gnrl_accuracy = []
                           )
            else:
                return dict(iter=[],
                            train_loss = [],
                            test_loss = [],
                            gnrl_loss = [],
                            train_depth_accuracy = [],
                            train_black_accuracy = [],
                            train_white_accuracy = [],
                            test_depth_accuracy = [],
                            test_black_accuracy = [],
                            test_white_accuracy = [],
                            gnrl_depth_accuracy = [],
                            gnrl_black_accuracy = [],
                            gnrl_white_accuracy = []
                           )
        elif self.testing_method =='supervised_decoder':
            return dict(iter=[],
                        train_recon_loss=[],
                        test_recon_loss =[],
                        gnrl_recon_loss =[],
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
            