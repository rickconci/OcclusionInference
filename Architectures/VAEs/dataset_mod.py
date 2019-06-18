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



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


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
    def __init__(self,image_paths, target_paths, image_size = 64 ):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_size = image_size
    def __getitem__(self, index):
        x_sample = default_loader(self.image_paths+ os.listdir(self.image_paths)[index])
        y_sample = default_loader(self.target_paths+os.listdir(self.target_paths)[index])

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        x = transform(x_sample)
        y = transform(y_sample)
        
        return x, y

    def __len__(self):
        return len(self.image_paths)