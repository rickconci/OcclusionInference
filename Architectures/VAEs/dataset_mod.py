import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class ImageData(Dataset):
    def __init__(self, FILENAME, mode='train'):
        # download pickle file of images
        images_dict = pickle.load(open( "{}/images_dict.p".format(FILENAME), "rb" ) )
        fulldatasize = int(images_dict["images"].shape[0]/2)
        training_size = int(fulldatasize*0.99)
        test_size = int(fulldatasize - training_size)
        self.mode = mode
        self.x_train = images_dict["images"][np.arange(0, 2*training_size, step=2),:,:,0]
        self.y_train = images_dict["images"][np.arange(1, 2*training_size, step=2),:,:,0]
        self.x_test = images_dict["images"][np.arange(2*training_size,2*fulldatasize, step=2),:,:,0]
        self.y_test = images_dict["images"][np.arange(2*training_size+1, 2*fulldatasize, step=2),:,:,0]
        
        self.x_train = self.x_train/self.x_train[0].max()
        self.y_train = self.x_train/self.y_train[0].max()
        self.x_test = self.x_train/self.x_test[0].max()
        self.y_test = self.x_train/self.y_test[0].max()


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index, mode='train'):
        if mode =='train':
            img = torch.from_numpy(self.x_train[index,:,:]).unsqueeze(0).float()
            label = torch.from_numpy(self.y_train[index,:,:]).unsqueeze(0).float()
        elif mode =='test':
            img = torch.from_numpy(self.x_test[index,:,:]).unsqueeze(0).float()
            label = torch.from_numpy(self.y_test[index,:,:]).unsqueeze(0).float()
        else:
            return(print("Incorrect mode: enter train or test"))

        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self, mode='train'):
        if mode =='train':
            return len(self.x_train)
        else:
            return len(self.x_test)
    