import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os
from scipy import stats
from torch.utils.data import Dataset, DataLoader
import operator
import matplotlib.pyplot as plt
import math
import pickle

output_dir = "/home/riccardo/Desktop/Experiments/Scatter_Plots_Decoders"

sys.path.insert(0, '/home/riccardo/Desktop/OcclusionInference/Architectures')
from models.BLT_models import multi_VAE, SB_decoder, spatial_broadcast_decoder
from data_loaders.dataset_sup import MyDataset_encoder, MyDataset_decoder
from solvers.losses import supervised_decoder_loss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

#%load_ext autoreload
#%autoreload 2

print("CREATING DATASET DICTIONARIES")


solid2_encoder  = {'name':'solid2_encoder', 'train_image_paths':'100k_2_digt_BW/digts/train/orig/', 'gnrl_image_paths':'100k_2_digt_BW/digts/gnrl/orig/', 'gnrl_target_paths':'100k_2_digt_BW/digts/digts_gnrl.csv'  }
solid2_decoder = {'name':'solid2_decoder' , 'train_image_paths': '100k_2_digt_BW/digts/train/orig/', 'gnrl_image_paths': '100k_2_digt_BW/digts/digts_gnrl.csv', 'gnrl_target_paths':'100k_2_digt_BW/digts/gnrl/orig/'}
border2_encoder = {'name':'border2_encoder'  , 'train_image_paths':'100k_2digt_BWE_2/digts/train/orig/','gnrl_image_paths': '100k_2digt_BWE_2/digts/gnrl/orig/', 'gnrl_target_paths':'100k_2digt_BWE_2/digts/digts_gnrl.csv' }
border2_decoder = {'name':'border2_decoder'  ,'train_image_paths':'100k_2digt_BWE_2/digts/train/orig/','gnrl_image_paths':'100k_2digt_BWE_2/digts/digts_gnrl.csv' , 'gnrl_target_paths':'100k_2digt_BWE_2/digts/gnrl/orig/' }
border3_encoder = {'name':'border3_encoder'   ,'train_image_paths':'100k_3digt_BWE/digts/train/orig/','gnrl_image_paths':'100k_3digt_BWE/digts/gnrl/orig/' , 'gnrl_target_paths':'100k_3digt_BWE/digts/digts_gnrl.csv' }
border3_decoder = {'name': 'border3_decoder'   ,'train_image_paths':'100k_3digt_BWE/digts/train/orig/','gnrl_image_paths':'100k_3digt_BWE/digts/digts_gnrl.csv' , 'gnrl_target_paths': '100k_3digt_BWE/digts/gnrl/orig/'}

#datasets = [border2_encoder, border2_decoder, border3_encoder,border3_decoder]

print("CREATING MODEL DICTIONARIES")


B_encoder_solid2 = {'encoder': 'B', 'decoder': 'B', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filenme': '/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bw/arch/B/main/last'}
B_matched_encoder_solid2 = {'encoder': 'B', 'decoder': 'B', 'n_filter': 35,  'kernel_size': 6, 'padding': 2, 'filenme':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bw/arch/B_matchedlr0_001/main/last'}
BL_encoder_solid2 = {'encoder': 'BL', 'decoder': 'B', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bw/arch/BL_lr0_001/main/last'}
BT_encoder_solid2 = {'encoder':'BT' , 'decoder':'B' , 'n_filter':32 ,  'kernel_size': 4, 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bw/arch/BT_lr0_001/main/last'}
BLT_encoder_solid2 = {'encoder':'BLT' , 'decoder':'B','n_filter': 32,  'kernel_size':4 , 'padding':1 , 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bw/arch/BLT/main/last'}



B_encoder_border2 ={'name' : 'B_encoder_border2', 'encoder': 'B', 'decoder': 'B', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bwe/arch/B_2/main/last'}
B_matched_encoder_border2 = {'name' : 'B_matched_encoder_border2', 'encoder': 'B', 'decoder': 'B', 'n_filter': 35,  'kernel_size': 6, 'padding': 2, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bwe/arch/B_matched_2/main/last'}
BL_encoder_border2 = {'name' :'BL_encoder_border2' , 'encoder': 'BL', 'decoder': 'B', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bwe/arch/BL_lr0_001_2/main/last'}
BT_encoder_border2 = {'name' : 'BT_encoder_border2','encoder':'BT' , 'decoder':'B' , 'n_filter':32 ,  'kernel_size': 4, 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bwe/arch/BT_lr0_001_2/main/20000'}
BLT_encoder_border2  = {'name' : 'BLT_encoder_border2','encoder':'BLT' , 'decoder':'B','n_filter': 32,  'kernel_size':4 , 'padding':1 , 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/2_digts/bwe/arch/BLT_lr0_001_2/main/last'}



B_encoder_border3 = {'name' : 'B_encoder_border3','encoder': 'B', 'decoder': 'BLT', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filename': '/home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/B_2/main/last'}
B_matched_encoder_border3  = {'name' : 'B_matched_encoder_border3','encoder': 'B', 'decoder': 'BLT', 'n_filter': 35,  'kernel_size': 6, 'padding': 2, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/B_matched_2/main/last'}
BL_encoder_border3  = {'name' : 'BL_encoder_border3','encoder': 'BL', 'decoder': 'BLT', 'n_filter': 32,  'kernel_size':4 , 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BL_lr0_001_2/main/last'}
BT_encoder_border3  = {'name' : 'BT_encoder_border3','encoder':'BT' , 'decoder':'BLT' , 'n_filter':32 ,  'kernel_size': 4, 'padding': 1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BT_2/main/last'}
BLT_encoder_border3  = {'name' : 'BLT_encoder_border3','encoder':'BLT' , 'decoder':'BLT','n_filter': 32,  'kernel_size':4 , 'padding':1, 'filename':'/home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BLT_2/main/last' }





# B_decoder_solid2 = {'encoder': 'B', 'decoder':'B' , 'n_filter': 32,  'kernel_size': 4, 'padding':1, 'filename': }
# B_matched_decoder_solid2 = {'encoder':'B' , 'decoder': 'B', 'n_filter':35 ,  'kernel_size':6 , 'padding':2 , 'filename': }
# BL_decoder_solid2 = {'encoder': 'B', 'decoder':'BL' , 'n_filter': 32,  'kernel_size':4 , 'padding':1 , 'filename': }
# BT_decoder_solid2 = {'encoder':'B' , 'decoder':'BT' , 'n_filter': 32,  'kernel_size': 4, 'padding':1 , 'filename': }
# BLT_decoder_solid2 = {'encoder': 'B', 'decoder':'BLT' , 'n_filter': 32,  'kernel_size': 4, 'padding': 1, 'filename': }


#comp loss w/ L2
B_decoder_border2 = {'name' : 'B_decoder_border2','encoder': 'B', 'decoder':'B' , 'n_filter': 32,  'kernel_size': 4, 'padding':1 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/loss_w_comp/B_lr_0_001/main/best_gnrl'}
B_matched_decoder_border2 = {'name' : 'B_matched_decoder_border2','encoder':'B' , 'decoder': 'B', 'n_filter':35 ,  'kernel_size':6 , 'padding':2 , 'filename':'/home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/loss_w_comp/B_matched_lr_0_005/main/best_gnrl' }
BL_decoder_border2 = {'name' : 'BL_decoder_border2','encoder': 'B', 'decoder':'BL' , 'n_filter': 32,  'kernel_size':4 , 'padding':1 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/loss_w_comp/BL_lr_0_005/main/best_gnrl'}
BT_decoder_border2 = {'name' : 'BT_decoder_border2','encoder':'B' , 'decoder':'BT' , 'n_filter': 32,  'kernel_size': 4, 'padding':1 , 'filename':'//home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/loss_w_comp/BT_lr_0_005/main/best_gnrl' }
BLT_decoder_border2 = {'name' : 'BLT_decoder_border2','encoder': 'B', 'decoder':'BLT' , 'n_filter': 32,  'kernel_size': 4, 'padding': 1, 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/loss_w_comp/BLT_lr_0_005/main/best_gnrl'}



B_decoder_border3 = {'name' : 'B_decoder_border3','encoder': 'B', 'decoder':'B' , 'n_filter': 32,  'kernel_size': 4, 'padding':1 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/3_digts/arch/B/main/last'}
B_matched_decoder_border3 = {'name' : 'B_matched_decoder_border3','encoder':'B' , 'decoder': 'B', 'n_filter':35 ,  'kernel_size':6 , 'padding':2 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/3_digts/arch/B_matched/main/last'}
BL_decoder_border3 = {'name' : 'BL_decoder_border3','encoder': 'B', 'decoder':'BL' , 'n_filter': 32,  'kernel_size':4 , 'padding':1 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/3_digts/arch/BL/main/last'}
BT_decoder_border3 = {'name' : 'BT_decoder_border3','encoder':'B' , 'decoder':'BT' , 'n_filter': 32,  'kernel_size': 4, 'padding':1 , 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/3_digts/arch/BT/main/last'}
BLT_decoder_border3 = {'name' : 'BLT_decoder_border3','encoder': 'B', 'decoder':'BLT' , 'n_filter': 32,  'kernel_size': 4, 'padding': 1, 'filename': '/home/riccardo/Desktop/Experiments/decoder_sup/3_digts/arch/BLT/main/last'}

print("SETTINGS:")

datasets = [border2_decoder]
nets = [B_decoder_border2, B_matched_decoder_border2, BL_decoder_border2, BT_decoder_border2, BLT_decoder_border2]
n_digits = 2
encoding_target_type = "depth_ordered_one_hot_xy"

print("datasets:", datasets )
print("nets:", nets)
print("n_digits:", n_digits)
print("encoding_target_type:", encoding_target_type)



print("CREATING RESPONSE MATRICES")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dset_dir = '/home/riccardo/Desktop/Data/'

Response_matrices = []
model_names = []
for dataset in datasets:
    print(dataset['name'])
    train_image_paths = dataset['train_image_paths']
    gnrl_image_paths = dataset['gnrl_image_paths'] 
    gnrl_target_paths = dataset['gnrl_target_paths']
    train_data_size = len(os.listdir(dset_dir+train_image_paths))
    if 'encoder' in dataset['name']:
        dset = MyDataset_encoder
    elif 'decoder' in dataset['name']:
        dset = MyDataset_decoder
    
    if 'solid2' in dataset['name']:
        n_digits = 2
        zdim = 24
    elif 'border2' in dataset['name']:
        n_digits = 2
        zdim = 24
    elif 'border3' in dataset['name']:
        n_digits = 3
        zdim = 36
    
    print(n_digits, zdim)
    gnrl_data = dset(dset_dir+gnrl_image_paths,dset_dir+gnrl_target_paths, 32, encoding_target_type, 'gnrl',train_data_size )
    gnrl_loader = DataLoader(gnrl_data,
                                batch_size=500,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True,
                                drop_last=False)
    
    
    for model in nets:
        model_names.append(model['name'])
        responses = torch.zeros( len(gnrl_data), 1)
        print(model['name'])
        net_1 = multi_VAE(model['encoder'],model['decoder'],zdim , 0 ,model['n_filter'] , 1, 4, False, model['kernel_size'], model['padding'], False)
        checkpoint = torch.load(model['filename'])
        net_1.load_state_dict(checkpoint['model_states']['net'])
        net_1.to(device)
        count = 0 
        with torch.no_grad():
            for sample in gnrl_loader:
                count +=1 
                x = sample['x'].to(device)
                trgt = sample['y'].to(device)
                
                output_1_list = net_1._decode(x)
                output_1 = output_1_list[-1]
                losses_1 = F.mse_loss(output_1, trgt)
                print(losses_1)
                
                responses[500*(count-1):500*(count), 0] = losses_1[:,0]
                    
                #print(torch.sum(responses[:,0]))
                
        Response_matrices.append(responses)
        
pickle.dump( Response_matrices, open( "{}/Response_matrices.p".format(output_dir), "wb" ) )

        
print("CALCULATING T VALUE AND P VALUE")
        
from itertools import combinations 

comb = combinations([0,1,2,3,4], 2) 
p_list = {}
for i,j in list(comb):
    mat1 = Response_matrices[i]
    mat2 = Response_matrices[j]
    name = model_names[i]+"~"+ model_names[j]
    
    x_mean = float(torch.mean(mat1))
    y_mean = float(torch.mean(mat2))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(mat1, mat2 ,s=0.1, c='black')
    ax.set_ylim(0,200)
    ax.set_xlim(0,200)
    ax.set_aspect('equal')
    ax.scatter(x_mean, y_mean, s=5, c='red')
    ax.plot([x_mean, x_mean],[y_mean, 0], 'm--',  linewidth=2 )
    ax.plot([0, x_mean],[y_mean, y_mean], 'b--',  linewidth=2 )
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    
    diff_vector = mat1 - mat2
    diff_mean = float(torch.mean(diff_vector))
    standard_dev = diff_vector.std(dim=0)
    standard_error = float(standard_dev/math.sqrt(2))
    t_test =   diff_mean/standard_error 
    t_test = float(t_test)
    p_value = stats.t.sf(np.abs(t_test), 1)*2

    p_list[name] = p_value
    
    ax.text(90, 180, 'diff_mean: {}'.format( np.round(diff_mean,2) ), fontsize=7)
    ax.text(90, 170, 'standard_error: {}'.format(np.round(standard_error,2)), fontsize=7)
    ax.text(90, 160, 't_stat: {}'.format(np.round(t_test,3)), fontsize=7)
    ax.text(90, 150, 'p_value: {}'.format(np.round(p_value,4)), fontsize=7)
    
    ax.text(90, 140, 'x_mean: {}'.format(np.round(x_mean,2)), fontsize=7, color='m')
    ax.text(90,130 , 'y_mean: {}'.format(np.round(y_mean,2)), fontsize=7, color='b')
    
    plt.xlabel(model_names[i])
    plt.ylabel(model_names[j])
    plt.title(name + "\n Reconstruction errors")
    plt.savefig("{}/{}".format(output_dir, name))
    
sorted_p_list = sorted(p_list.items(), key=operator.itemgetter(1))
print(sorted_p_list)
