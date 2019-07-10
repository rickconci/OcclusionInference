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
        self.x[0,:,:][self.x[0,:,:] == torch.median(self.x[0,:,:])] = 0.5
        self.y[0,:,:][self.y[0,:,:] == torch.median(self.y[0,:,:])] = 0.5

        sample = {'x':self.x, 'y':self.y}
        return sample
        #return self.x, self.y

    def __len__(self):
        return len(os.listdir(self.image_paths))

class MyDataset_decoder(Dataset):
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
        self.x[0,:,:][self.x[0,:,:] == torch.median(self.x[0,:,:])] = 0.5
        self.y[0,:,:][self.y[0,:,:] == torch.median(self.y[0,:,:])] = 0.5

        sample = {'x':self.x, 'y':self.y}
        return sample
        #return self.x, self.y

    def __len__(self):
        return len(os.listdir(self.image_paths))
    
    
class MyDataset_encoder(Dataset):
    def __init__(self,image_paths, target_paths, image_size, encoder_target_type, train_test):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_size = image_size
        
        targets = one_hot_targets(csv_path=target_paths, train_test_type=train_test)
        
        if encoder_target_type=='joint':
            self.digt_targets = targets.joint_targets()
        elif encoder_target_type=='black_white':
            self.digt_targets = targets.depth_black_white_one_hot(depth=False, xy=False)
        elif encoder_target_type=='depth_black_white':
            self.digt_targets = targets.depth_black_white_one_hot(depth=True, xy=False)
            #print(type(self.digt_targets))
        elif encoder_target_type=='depth_black_white_xy_xy':
            self.digt_targets = targets.depth_black_white_one_hot(depth=True, xy=True)
            #print(type(self.digt_targets))
        else:
            raise NotImplementedError('encoder type not correct')
        
            
    def __getitem__(self, index):
        x_sample = default_loader(self.image_paths+ sorted(os.listdir(self.image_paths))[index])
        #print(type(self.digt_targets))
        #print(self.digt_targets.shape)
        y_sample = self.digt_targets[index,:]

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        
        x = transform(x_sample)
        
        x[0,:,:][x[0,:,:] == torch.median(x[0,:,:])] = 0.5
        x[0,:,:][x[0,:,:] == torch.max(x[0,:,:])] = 0

        sample = {'x':x, 'y':y_sample}
        return sample
        #return self.x, self.y

    def __len__(self):
        return len(os.listdir(self.image_paths))


    
class one_hot_targets():
    def __init__(self, csv_path, train_test_type):
        #file = '/Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/digts.csv'
        data = pd.read_csv(csv_path, header=None)
        self.train_data = data.iloc[:9800, :]
        self.test_data = data.iloc[9800:, :]
        self.train_test_type = train_test_type
        #print(self.train_test_type)

        if train_test_type=='train':
            self.digt_list_train = self.train_data.iloc[:, [5,21]].values.astype(int)
            cols_train = self.train_data.iloc[:, [12, 32]].values/255
            self.cols_train = cols_train.astype(int)
            self.x_back = self.train_data.iloc[:, 6].values.astype(float)
            self.y_back = self.train_data.iloc[:, 7].values.astype(float)
            self.x_front = self.train_data.iloc[:, 22].values.astype(float)
            self.y_front = self.train_data.iloc[:, 23].values.astype(float)
        elif train_test_type=='test':
            self.digt_list_test = self.test_data.iloc[:, [5,21]].values.astype(int)
            cols_test = self.test_data.iloc[:, [12, 32]].values/255
            self.cols_test = cols_test.astype(int)
            self.x_back = self.test_data.iloc[:, 6].values.astype(float)
            self.y_back= self.test_data.iloc[:, 7].values.astype(float)
            self.x_front = self.test_data.iloc[:, 22].values.astype(float)
            self.y_front = self.test_data.iloc[:, 23].values.astype(float)
       
        self.distance = np.sqrt((self.x_front - self.x_back)**2 + (self.y_front - self.y_back)**2)
    
    def joint_targets(self):
        if self.train_test_type=='train':
            back = self.digt_list_train[:,0]
            front =self.digt_list_train[:,1]
            col_back = torch.LongTensor(self.cols_train[:,0]).view(9800, -1)
        elif self.train_test_type=='test':
            back = self.digt_list_test[:,0]
            front =self.digt_list_test[:,1]
            col_back = torch.LongTensor(self.cols_test[:,0]).view(200, -1)
        
        n_values = np.max(back) + 1
        back_one_hot = torch.LongTensor(np.eye(n_values)[back])
        front_one_hot = torch.LongTensor(np.eye(n_values)[front])
        back_front_one_hot = torch.cat((back_one_hot,front_one_hot),1)
        col_back_front_one_hot = torch.cat((col_back, back_front_one_hot),1)
            
        joint_digts_id = back_one_hot + front_one_hot
        return(joint_digts_id)
      
    def depth_black_white_one_hot(self, depth, xy):
        if self.train_test_type=='train':
            new_df = np.zeros((self.train_data.shape[0], 2))
            for i in range(self.train_data.shape[0]):
                new_df[i,self.cols_train[i,0]] = self.digt_list_train[i,0]
                new_df[i,self.cols_train[i,1]] = self.digt_list_train[i,1]
            depth_black = torch.FloatTensor(self.cols_train[:,0]).view(9800, -1)
        elif self.train_test_type=='test':
            new_df = np.zeros((self.test_data.shape[0], 2))
            for i in range(self.test_data.shape[0]):
                new_df[i,self.cols_test[i,0]] = self.digt_list_test[i,0]
                new_df[i,self.cols_test[i,1]] = self.digt_list_test[i,1]
            depth_black = torch.FloatTensor(self.cols_test[:,0]).view(200, -1)
            
        black = new_df[:,0].astype(int)
        black = new_df[:,0].astype(int)
        white = new_df[:,1].astype(int)
        n_values = np.max(black) + 1
        black_one_hot = torch.FloatTensor(np.eye(n_values)[black])
        white_one_hot = torch.FloatTensor(np.eye(n_values)[white])
        black_white_one_hot = torch.cat((black_one_hot,white_one_hot), 1)
        if depth==True:
            if xy==True:
                self.x_back = torch.from_numpy(self.x_back).float().unsqueeze(1)
                self.y_back = torch.from_numpy(self.y_back).float().unsqueeze(1)
                self.x_front = torch.from_numpy(self.x_front).float().unsqueeze(1)
                self.y_front = torch.from_numpy(self.y_front).float().unsqueeze(1)
                
                depth_black_white_one_hot = torch.cat((depth_black, black_white_one_hot),1)
                
                depth_black_white_one_hot_xy_xy = torch.cat((depth_black_white_one_hot,
                                                       self.x_back, self.y_back,
                                                       self.x_front,self.y_front),1)
                #print("TYPEE", type(depth_black_white_one_hot))
                return(depth_black_white_one_hot_xy_xy)
            else:
                depth_black_white_one_hot = torch.cat((depth_black, black_white_one_hot),1)
                return(depth_black_white_one_hot)
        else:
            return(black_white_one_hot)
        

    
def return_data_unsupervised(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    
    #assert image_size == 64, 'currently only image size of 64 is supported'
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
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_image_paths = os.path.join(dset_dir + "test/orig/")
    test_target_paths = os.path.join(dset_dir + "test/inverse/")
    dset_test= MyDataset
    test_kwargs = {'image_paths': test_image_paths,
                    'target_paths': test_target_paths,
                    'image_size': image_size}
    test_data = dset_test(**test_kwargs)
    test_loader = DataLoader(test_data,
                              batch_size=200,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)
    print('{} train images, {} test images"'.format(train_data.__len__(), test_data.__len__()))
    
    gnrl_image_paths = os.path.join(dset_dir + "gnrl/orig/")
    gnrl_target_paths = os.path.join(dset_dir + "gnrl/inverse/")
    dset_gnrl = MyDataset
    gnrl_kwargs = {'image_paths':gnrl_image_paths,
                    'target_paths': gnrl_target_paths,
                    'image_size': image_size}
    gnrl_data = dset_gnrl(**gnrl_kwargs) 
    gnrl_loader = DataLoader(gnrl_data,
                              batch_size=200,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    return unsup_train_loader, unsup_test_loader, unsup_gnrl_loader

def return_data_sup_encoder(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 32
    encoder_target_type = args.encoder_target_type
    print("encoding:",encoder_target_type )
    
    train_image_paths = "{}train/orig/".format(dset_dir)
    train_target_paths = "{}digts.csv".format(dset_dir)


    dset_train = MyDataset_encoder
    train_kwargs = {'image_paths':train_image_paths,
                    'target_paths': train_target_paths,
                    'image_size': image_size, 
                   'encoder_target_type':encoder_target_type,
                   'train_test': 'train'}

    train_data = dset_train(**train_kwargs) 
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                            drop_last=False)

    test_image_paths = os.path.join(dset_dir + "test/orig/")
    test_target_paths = "{}digts.csv".format(dset_dir)
    dset_test= MyDataset_encoder
    test_kwargs = {'image_paths':test_image_paths,
                    'target_paths': test_target_paths,
                    'image_size': image_size, 
                   'encoder_target_type':encoder_target_type,
                   'train_test': 'test'}
    test_data = dset_test(**test_kwargs)
    test_loader = DataLoader(test_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)
    print('{} train images, {} test images"'.format(train_data.__len__(), test_data.__len__()))


    return train_loader, test_loader


def return_data_sup_decoder(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size