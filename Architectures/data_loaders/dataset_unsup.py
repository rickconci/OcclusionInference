import os
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms

import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL
import pandas as pd 

#https://discuss.pytorch.org/t/custom-image-dataset-for-autoencoder/16118/2
#https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7

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
    
class MyDataset_unsup(Dataset):
    def __init__(self,image_paths, target_paths, image_size, train_test_gnrl, train_data_size, test_data_size):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_size = image_size
        
        if train_test_gnrl == 'train':
            print("Sorting train image files")
            self.files_img = [image_paths+ 'orig_{}.png'.format(i) for i in range(0,train_data_size)]
            self.files_tgt = [target_paths+ 'inverse_{}.png'.format(i) for i in range(0,train_data_size)]
        elif train_test_gnrl == 'test':
            print("Sorting test image files")
            tot_data_size = train_data_size + test_data_size
            self.files_img = [image_paths+ 'orig_{}.bmp'.format(i) for i in range(train_data_size, tot_data_size)] 
            self.files_tgt = [target_paths+ 'inverse_{}.png'.format(i) for i in range(train_data_size, tot_data_size)]
        elif train_test_gnrl == 'gnrl':
            gnrl_data_size = len(os.listdir(image_paths))
            print("Sorting gnrl image files")
            self.files_img = [image_paths+ 'orig_{}.bmp'.format(i) for i in range(0,gnrl_data_size)]
            self.files_tgt = [target_paths+ 'inverse_{}.png'.format(i) for i in range(0,gnrl_data_size)]
        
    def __getitem__(self, index):
        
        x_sample = default_loader(self.files_img[index])
        y_sample = default_loader(self.files_tgt[index])

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        self.x = transform(x_sample)
        self.y = transform(y_sample)
        self.x[0,:,:][self.x[0,:,:] == torch.median(self.x[0,:,:])] = 0.5
        self.y[0,:,:][self.y[0,:,:] == torch.median(self.y[0,:,:])] = 0.5

        sample = {'x':self.x, 'y':self.y}
        return sample
        #return self.x, self.y

    def __len__(self):
        return len(os.listdir(self.image_paths))

    
def return_data_unsupervised(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    
    train_image_paths = "{}train/orig/".format(dset_dir)
    train_target_paths = "{}train/inverse/".format(dset_dir)
    test_image_paths = os.path.join(dset_dir + "test/orig/")
    test_target_paths = os.path.join(dset_dir + "test/inverse/")
    
    train_data_size = len(os.listdir(train_image_paths))
    test_data_size = len(os.listdir(test_image_paths))
    
    dset_train = MyDataset_unsup
    train_kwargs = {'image_paths':train_image_paths,
                    'target_paths': train_target_paths,
                    'image_size': image_size,
                    'train_test_gnrl': 'train',
                   'train_data_size':train_data_size,
                   'test_data_size': test_data_size}
    train_data = dset_train(**train_kwargs) 
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    
    dset_test= MyDataset_unsup
    test_kwargs = {'image_paths': test_image_paths,
                    'target_paths': test_target_paths,
                    'image_size': image_size,
                   'train_test_gnrl': 'test',
                   'train_data_size':train_data_size,
                   'test_data_size': test_data_size}
    test_data = dset_test(**test_kwargs)
    test_loader = DataLoader(test_data,
                              batch_size=200,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)
    
    gnrl_image_paths = os.path.join(dset_dir + "gnrl/orig/")
    gnrl_target_paths = os.path.join(dset_dir + "gnrl/inverse/")
    if os.path.exists(gnrl_image_paths):
        dset_gnrl = MyDataset_unsup
        gnrl_kwargs = {'image_paths':gnrl_image_paths,
                        'target_paths': gnrl_target_paths,
                        'image_size': image_size,
                       'train_test_gnrl':'gnrl',
                       'train_data_size':train_data_size,
                       'test_data_size': test_data_size}
        gnrl_data = dset_gnrl(**gnrl_kwargs) 
        gnrl_loader = DataLoader(gnrl_data,
                                  batch_size=200,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)
    
        gnrl_data_size = len(os.listdir(gnrl_image_paths))
        
        print('{} train images, {} test images {} generalisation images"'.format(
            train_data_size, test_data_size, gnrl_data_size))
    else:
        gnrl_loader = 0
        gnrl_data = 0
        print('{} train images, {} test images"'.format(
            train_data_size, test_data_size))

    
    return train_loader, test_loader, gnrl_loader, test_data, gnrl_data
   
