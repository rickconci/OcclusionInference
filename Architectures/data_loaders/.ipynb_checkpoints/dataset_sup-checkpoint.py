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
    

class MyDataset_encoder(Dataset):
    def __init__(self,image_paths, target_paths, image_size, encoder_target_type, train_test_gnrl, train_data_size):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_size = image_size
        self.train_test_gnrl = train_test_gnrl
        self.train_data_size = train_data_size
        if train_test_gnrl == 'train':
            print("Sorting train image files")
            self.files_img = [image_paths+ 'orig_{}.png'.format(i) for i in range(0,train_data_size)]       
        elif train_test_gnrl == 'gnrl':
            self.gnrl_data_size = len(os.listdir(image_paths))
            print("Sorting gnrl image files")
            self.files_img = [image_paths+ 'orig_{}.png'.format(i) for i in range(0,self.gnrl_data_size)]
        
        targets = one_hot_targets(target_paths,train_test_gnrl)
        
        if encoder_target_type=='joint':
            self.digit_targets = targets.joint_targets()
        elif encoder_target_type=='black_white':
            self.digit_targets = targets.depth_black_white_one_hot(depth=False, xy=False)
        elif encoder_target_type=='depth_black_white':
            self.digit_targets = targets.depth_black_white_one_hot(depth=True, xy=False)
            #print(type(self.digt_targets))
        elif encoder_target_type=='depth_black_white_xy_xy':
            self.digit_targets = targets.depth_black_white_one_hot(depth=True, xy=True)
            #print(type(self.digt_targets))
        elif encoder_target_type =='depth_ordered_one_hot':
            self.digit_targets = targets.depth_ordered_one_hot(xy=False)
        elif encoder_target_type =='depth_ordered_one_hot_xy':
            self.digit_targets = targets.depth_ordered_one_hot(xy=True)
        else:
            raise NotImplementedError('encoder type not correct')
       
        
            
    def __getitem__(self, index):
        x_sample = default_loader(self.files_img[index])
        y_sample = self.digit_targets[index,:]

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        
        x = transform(x_sample)
        
        x[0,:,:][x[0,:,:] == torch.median(x[0,:,:])] = 0.5

        sample = {'x':x, 'y':y_sample}
        return sample

    def __len__(self):
        if self.train_test_gnrl == 'train':
            return self.train_data_size
        elif self.train_test_gnrl == 'gnrl':
            return self.gnrl_data_size
    
        

class MyDataset_decoder(Dataset):
    def __init__(self,x_paths, y_paths, image_size , encoder_target_type, train_test_gnrl, train_data_size,):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.image_size = image_size
        self.train_data_size = train_data_size
        self.train_test_gnrl = train_test_gnrl
        
        if train_test_gnrl == 'train':
            print("Sorting train image files")
            self.files_img = [y_paths+ 'orig_{}.png'.format(i) for i in range(0,train_data_size)]       
        elif train_test_gnrl == 'gnrl':
            self.gnrl_data_size = len(os.listdir(y_paths))
            print("Sorting gnrl image files")
            self.files_img = [y_paths+ 'orig_{}.png'.format(i) for i in range(0,self.gnrl_data_size)]
        
        inputs = one_hot_targets(x_paths,train_test_gnrl)
        
        if encoder_target_type=='joint':
            self.digit_targets = inputs.joint_targets()
        elif encoder_target_type=='black_white':
            self.digit_targets = inputs.depth_black_white_one_hot(depth=False, xy=False)
        elif encoder_target_type=='depth_black_white':
            self.digit_targets = inputs.depth_black_white_one_hot(depth=True, xy=False)
        elif encoder_target_type=='depth_black_white_xy_xy':
            self.digit_targets = inputs.depth_black_white_one_hot(depth=True, xy=True)
        elif encoder_target_type =='depth_ordered_one_hot':
            self.digit_targets = inputs.depth_ordered_one_hot(xy=False)
        elif encoder_target_type =='depth_ordered_one_hot_xy':
            self.digit_targets = inputs.depth_ordered_one_hot(xy=True)
        else:
            raise NotImplementedError('encoder type not correct')
        
    def __getitem__(self, index):
        
        x_sample = self.digit_targets[index,:] 
        y_sample = default_loader(self.files_img[index])

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(),
            transforms.ToTensor(),])
        
        y = transform(y_sample)
        y[0,:,:][y[0,:,:] == torch.median(y[0,:,:])] = 0.5

        sample = {'x':x_sample, 'y':y}
        return sample
        #return self.x, self.y

    def __len__(self):
        if self.train_test_gnrl == 'train':
            return self.train_data_size
        elif self.train_test_gnrl == 'gnrl':
            return self.gnrl_data_size
    
    
    
class one_hot_targets():
    def __init__(self, csv_path, train_test_type):
        #file = '/Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/digts.csv'
        self.data = pd.read_csv(csv_path, header=None)
        
        if train_test_type == 'gnrl':
            self.gnrl_data= self.data
            self.gnrl_data_size = self.data.shape[0]
        elif train_test_type == 'train':
            self.train_data = self.data
            self.train_data_size = self.train_data.shape[0]
        
        self.train_test_type = train_test_type

        #print(self.train_test_type)
        if self.data.shape[1] > 37  :
            if train_test_type=='train':
                self.digt_list_train = self.train_data.iloc[:, [5,21, 37]].values.astype(int)
                self.x_back = self.train_data.iloc[:, 6].values.astype(float)
                self.y_back = self.train_data.iloc[:, 7].values.astype(float)
                self.x_mid = self.train_data.iloc[:, 22].values.astype(float)
                self.y_mid = self.train_data.iloc[:, 23].values.astype(float)
                self.x_front = self.train_data.iloc[:, 38].values.astype(float)
                self.y_front = self.train_data.iloc[:, 39].values.astype(float)
            elif train_test_type == 'gnrl':
                self.digt_list_gnrl = self.gnrl_data.iloc[:, [5,21, 37]].values.astype(int)
                self.x_back = self.gnrl_data.iloc[:, 6].values.astype(float)
                self.y_back = self.gnrl_data.iloc[:, 7].values.astype(float)
                self.x_mid = self.gnrl_data.iloc[:, 22].values.astype(float)
                self.y_mid = self.gnrl_data.iloc[:, 23].values.astype(float)
                self.x_front = self.gnrl_data.iloc[:, 38].values.astype(float)
                self.y_front = self.gnrl_data.iloc[:, 39].values.astype(float)
        elif self.data.shape[1] <= 37:
            if train_test_type=='train':
                self.digt_list_train = self.train_data.iloc[:, [5,21]].values.astype(int)
                cols_train = self.train_data.iloc[:, [12, 32]].values/255
                self.cols_train = cols_train.astype(int)
                self.x_back = self.train_data.iloc[:, 6].values.astype(float)
                self.y_back = self.train_data.iloc[:, 7].values.astype(float)
                self.x_front = self.train_data.iloc[:, 22].values.astype(float)
                self.y_front = self.train_data.iloc[:, 23].values.astype(float)
            elif train_test_type=='gnrl':
                self.digt_list_gnrl = self.gnrl_data.iloc[:, [5,21]].values.astype(int)
                cols_gnrl = self.gnrl_data.iloc[:, [12, 32]].values/255
                self.cols_gnrl = cols_gnrl.astype(int)
                self.x_back = self.gnrl_data.iloc[:, 6].values.astype(float)
                self.y_back= self.gnrl_data.iloc[:, 7].values.astype(float)
                self.x_front = self.gnrl_data.iloc[:, 22].values.astype(float)
                self.y_front = self.gnrl_data.iloc[:, 23].values.astype(float)
       
        #self.distance = np.sqrt((self.x_front - self.x_back)**2 + (self.y_front - self.y_back)**2)
        #plt.hist(self.distance, bins=30)
        #plt.ylabel('Probability');
        #plt.savefig('Distances histogram.png')
        
    def joint_targets(self):
        if self.data.shape[1] > 37:
            back = torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,0]])
            mid =  torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,1]])
            front =  torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,2]])
            
            return(back+mid+front)
        elif self.data.shape[1] <= 37:
            if self.train_test_type=='train':
                back = self.digt_list_train[:,0]
                front =self.digt_list_train[:,1]
                col_back = torch.LongTensor(self.cols_train[:,0]).view(self.train_data_size, -1)
            elif self.train_test_type =='gnrl':
                back = self.digt_list_gnrl[:,0]
                front =self.digt_list_gnrl[:,1]
                col_back = torch.LongTensor(self.cols_gnrl[:,0]).view(self.gnrl_data_size, -1)
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
            depth_black = torch.FloatTensor(self.cols_train[:,0]).view(self.train_data_size, -1)
        elif self.train_test_type=='gnrl':
            new_df = np.zeros((self.gnrl_data.shape[0], 2))
            for i in range(self.gnrl_data.shape[0]):
                new_df[i,self.cols_gnrl[i,0]] = self.digt_list_gnrl[i,0]
                new_df[i,self.cols_gnrl[i,1]] = self.digt_list_gnrl[i,1]
            depth_black = torch.FloatTensor(self.cols_gnrl[:,0]).view(self.gnrl_data_size, -1)
            
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
        
    def depth_ordered_one_hot(self, xy):
        if self.data.shape[1] > 37:
            if self.train_test_type=='train':
                n_values = 10
                back = torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,0]])
                mid =  torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,1]])
                front =  torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,2]])
            elif self.train_test_type=='gnrl':
                n_values = 10
                back = torch.FloatTensor(np.eye(n_values)[self.digt_list_gnrl[:,0]])
                mid =  torch.FloatTensor(np.eye(n_values)[self.digt_list_gnrl[:,1]])
                front =  torch.FloatTensor(np.eye(n_values)[self.digt_list_gnrl[:,2]])
            if xy==True:
                self.x_back = torch.from_numpy(self.x_back).float().unsqueeze(1)
                self.y_back = torch.from_numpy(self.y_back).float().unsqueeze(1)
                self.x_mid = torch.from_numpy(self.x_mid).float().unsqueeze(1)
                self.y_mid = torch.from_numpy(self.y_mid).float().unsqueeze(1)
                self.x_front = torch.from_numpy(self.x_front).float().unsqueeze(1)
                self.y_front = torch.from_numpy(self.y_front).float().unsqueeze(1)
                output = torch.cat((back,mid,front, self.x_back, self.y_back, self.x_mid, self.y_mid, self.x_front,self.y_front),1)
            else:
                output = torch.cat((back,mid,front),1)
            return(output)
        elif self.data.shape[1] <= 37:
            if self.train_test_type=='train':
                n_values = 10
                back = torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,0]])
                front =  torch.FloatTensor(np.eye(n_values)[self.digt_list_train[:,1]])
            elif self.train_test_type=='gnrl':
                n_values = 10
                back = torch.FloatTensor(np.eye(n_values)[self.digt_list_gnrl[:,0]])
                front =  torch.FloatTensor(np.eye(n_values)[self.digt_list_gnrl[:,1]])
            if xy==True:
                self.x_back = torch.from_numpy(self.x_back).float().unsqueeze(1)
                self.y_back = torch.from_numpy(self.y_back).float().unsqueeze(1)
                self.x_front = torch.from_numpy(self.x_front).float().unsqueeze(1)
                self.y_front = torch.from_numpy(self.y_front).float().unsqueeze(1)
                output = torch.cat((back, front, self.x_back, self.y_back,self.x_front,self.y_front),1)
            else:
                output = torch.cat((back,front),1)
            return(output)
            
    
def return_data_sup_encoder(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    output_dir =args.output_dir
    assert image_size == 32
    encoder_target_type = args.encoder_target_type
    print("encoding:",encoder_target_type )
    
   
    
    train_image_paths = "{}train/orig/".format(dset_dir)
    train_target_paths = "{}digts.csv".format(dset_dir)
 
    train_data_size = len(os.listdir(train_image_paths))
    
    train_data_size = 0
    for file in os.listdir(train_image_paths):
        if file.endswith(".png"):
            train_data_size +=1
    
    dset_train = MyDataset_encoder
    train_kwargs = {'image_paths':train_image_paths,
                    'target_paths': train_target_paths,
                    'image_size': image_size, 
                   'encoder_target_type':encoder_target_type,
                   'train_test_gnrl': 'train',
                    'train_data_size': train_data_size}

    train_data = dset_train(**train_kwargs) 
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                            drop_last=False)
   

    gnrl_image_paths = os.path.join(dset_dir + "gnrl/orig/")
    gnrl_target_paths = "{}digts_gnrl.csv".format(dset_dir)
    if os.path.exists(gnrl_image_paths):
        gnrl_data_size = len(os.listdir(gnrl_image_paths))
        
        
        dset_gnrl = MyDataset_encoder
        gnrl_kwargs = {'image_paths':gnrl_image_paths,
                        'target_paths': gnrl_target_paths,
                        'image_size': image_size, 
                       'encoder_target_type':encoder_target_type,
                       'train_test_gnrl': 'gnrl',
                        'train_data_size': train_data_size}

        gnrl_data = dset_gnrl(**gnrl_kwargs) 
        gnrl_loader = DataLoader(gnrl_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                drop_last=False)

        
        
        print('{} train images, {} generalisation images"'.format(
            train_data_size, gnrl_data_size))
    else:
         print('{} train images"'.format(
            train_data_size))

    return train_loader, gnrl_loader


def return_data_sup_decoder(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 32
    encoder_target_type = args.encoder_target_type

    
    x_train_paths = "{}digts.csv".format(dset_dir)
    y_train_paths = "{}train/orig/".format(dset_dir) 

    train_data_size = len(os.listdir(y_train_paths))
    
    train_data_size = 0
    for file in os.listdir(y_train_paths):
        if file.endswith(".png"):
            train_data_size +=1
    #print(train_data_size)
    
    
    dset_train = MyDataset_decoder
    train_kwargs = {'x_paths':x_train_paths,
                    'y_paths': y_train_paths,
                    'image_size': image_size, 
                   'encoder_target_type':encoder_target_type,
                   'train_test_gnrl': 'train',
                    'train_data_size': train_data_size}

    train_data = dset_train(**train_kwargs) 
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                            drop_last=False)
   

    x_gnrl_paths = "{}digts_gnrl.csv".format(dset_dir)
    y_gnrl_paths = os.path.join(dset_dir + "gnrl/orig/")
    
    if os.path.exists(y_gnrl_paths):
        
        dset_gnrl = MyDataset_decoder
        gnrl_kwargs = {'x_paths':x_gnrl_paths,
                       'y_paths': y_gnrl_paths,
                        'image_size': image_size,
                       'encoder_target_type':encoder_target_type,
                       'train_test_gnrl':'gnrl',
                       'train_data_size':train_data_size}
        gnrl_data = dset_gnrl(**gnrl_kwargs) 
        gnrl_loader = DataLoader(gnrl_data,
                                  batch_size=200,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)
    
        gnrl_data_size = len(os.listdir(y_gnrl_paths))
        
        print('{} train images,  {} generalisation images"'.format(
            train_data_size, gnrl_data_size))
    else:
        gnrl_loader = 0
        gnrl_data = 0
        print('{} train images"'.format(
            train_data_size))

    return train_loader, gnrl_loader, gnrl_data
