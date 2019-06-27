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
    
class MyDataset(Dataset):
    def __init__(self,image_paths, target_paths, image_size ):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_size = image_size
    def __getitem__(self, index):
        x_sample = default_loader(self.image_paths+ sorted(os.listdir(self.image_paths))[index])
        y_sample = default_loader(self.target_paths+ sorted(os.listdir(self.target_paths))[index])

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        self.x = transform(x_sample)
        self.y = transform(y_sample)
        sample = {'x':self.x, 'y':self.y}
        return sample
        #return self.x, self.y

    def __len__(self):
        return len(os.listdir(self.image_paths))
    
    
def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    
    #assert image_size == 64, 'currently only image size of 64 is supported'
    print(dset_dir)
    train_image_paths = "{}train/orig/".format(dset_dir)
    train_target_paths = "{}train/inverse/".format(dset_dir)
    print("train_image_paths: {}".format(train_image_paths))
    dset_train = MyDataset
    train_kwargs = {'image_paths':train_image_paths,
                    'target_paths': train_target_paths,
                    'image_size': image_size}
    train_data = dset_train(**train_kwargs) 
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_image_paths = os.path.join(dset_dir + "test/orig/")
    test_target_paths = os.path.join(dset_dir + "test/inverse/")
    dset_test= MyDataset
    test_kwargs = {'image_paths': test_image_paths,
                    'target_paths': test_target_paths,
                    'image_size': image_size}
    test_data = dset_test(**train_kwargs) 
    test_loader = DataLoader(test_data,
                              batch_size=200,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)
    
    
    
    return train_loader, test_loader
    