import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision
import shutil
from tqdm import tqdm
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/riccardoconci/Desktop/code/ZuckermanProject/OcclusionInference/Architectures')
#sys.path.insert(0, '/home/riccardo/Desktop/OcclusionInference/Architectures')

from models.BLT_models import SB_decoder, spatial_broadcast_decoder

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.rcParams['figure.figsize'] = [15, 15]

#from model_BLT_VAE import reparametrize_gaussian, reparametrize_bernoulli



def traverse_z(NN, example_id, ID, output_dir, global_iter, sbd ,num_frames = 100 ):
    z_dim = NN.z_dim_tot
    
    x_test_sample = example_id['x']
    y_test_sample = example_id['y']
    x_test_sample = torch.unsqueeze(x_test_sample, 0)
    
    #encode a sample image
    z_distributions = NN._encode(x_test_sample)
    if NN.z_dim_bern == 0:
        z_sample = z_distributions[:, :NN.z_dim_gauss]
    elif NN.z_dim_gauss == 0:
        z_sample = z_distributions[:, :NN.z_dim_bern]
    elif NN.z_dim_bern !=0 and NN.z_dim_gauss != 0:
        p = z_distributions[:, :NN.z_dim_bern]
        mu = z_distributions[:, NN.z_dim_bern:(NN.z_dim_bern+NN.z_dim_gauss)]
        z_sample = torch.cat((p, mu),1) 
   
    num_slice = int(1000/num_frames)

    if NN.z_dim_bern == 0:
        #create sorted normal samples & transverse_input matrix made from z encodings of sample image
        dist_samples = np.random.normal(loc=0, scale=1, size=1000)
        dist_samples.sort()
        dist_samples = torch.from_numpy(dist_samples[0::num_slice])
    elif NN.z_dim_gauss == 0:
        dist_samples = np.random.uniform(low=0, high=1, size=1000)
        dist_samples.sort()
        dist_samples = torch.from_numpy(dist_samples[0::num_slice])
    elif NN.z_dim_bern !=0 and NN.z_dim_gauss != 0:
        dist_samples_1= np.random.uniform(low=0, high=1, size=1000)
        dist_samples_2 = np.random.normal(loc=0, scale=1, size=1000)
        dist_samples_1.sort()
        dist_samples_2.sort()
        dist_samples_1 = torch.from_numpy(dist_samples_1[0::num_slice])
        dist_samples_2 = torch.from_numpy(dist_samples_2[0::num_slice])
        
            
    traverse_input = torch.mul(torch.ones(num_frames*z_dim,1),z_sample)
    
     
    #print(traverse_input.shape)

    if NN.z_dim_bern !=0 and NN.z_dim_gauss != 0:
        indexs = np.arange(0, num_frames*z_dim, num_frames)
        for i in indexs:
            z = int(i/num_frames)
            if z <= NN.z_dim_bern:
                traverse_input[i:(i+num_frames),z] = dist_samples_1
            else:
                traverse_input[i:(i+num_frames),z] = dist_samples_2
    else:
         #Populate matrix with individually varying Zs
        indexs = np.arange(0, num_frames*z_dim, num_frames)
        for i in indexs:
            z = int(i/num_frames)
            traverse_input[i:(i+num_frames),z] = dist_samples
   
    if sbd:
        sbd_decoder = SB_decoder(NN.z_dim_bern, NN.z_dim_gauss, NN.n_filter, NN.nc)
        sbd_model = spatial_broadcast_decoder()
        z_sample = sbd_model(z_sample)
        x_recon = sbd_decoder(z_sample)
        traverse_input = sbd_model(traverse_input)
        reconst = sbd_decoder(traverse_input)
        print(reconst.shape)
    else:
        #create all reconstruction images
        x_recon = NN._decode(z_sample)
        reconst = NN._decode(traverse_input)
        print(reconst.shape)

    #Create GIFs
    indexs = np.arange(0, num_frames*z_dim, num_frames)
    for i in indexs:
        #save images for each gif into the images list
        images = []
        for e in range(num_frames):
            #save images to make gifs into different folders
            filename = '{}/traversals{}_{}/z{}/img{}.png'.format(output_dir,global_iter,ID,int(i/num_frames),e)
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torchvision.utils.save_image(F.sigmoid(reconst[i+e,0,:,:].cpu()) , filename)
            images.append(imageio.imread(filename))


        #save all gifs into same folder
        filename_2 = '{}/traversals_gifs{}_{}/traversing_z_{}.gif'.format(
            output_dir,global_iter, ID,int(i/num_frames),int(i/num_frames))
        directory_2 = os.path.dirname(filename_2)
        if not os.path.exists(directory_2):
                os.makedirs(directory_2)
        imageio.mimsave('{}/traversals_gifs{}_{}/traversing_z_{}.gif'.format(
            output_dir, global_iter, ID, int(i/num_frames),int(i/num_frames)), images)
        
        with open('{}/traversals_gifs{}_{}/encoded_z.txt'.format(output_dir,global_iter,ID), 'w') as f:
            f.write(str(z_sample.numpy()))
        
        #add the reconstruction image to the GIF image folder
        torchvision.utils.save_image(F.sigmoid(x_recon[0,0,:,:]),
                                        '{}/traversals_gifs{}_{}/recon.png'.format(output_dir,global_iter,ID))
        #add the actual target image to the GIF image folder
        torchvision.utils.save_image(y_test_sample[0,:,:],
                                        '{}/traversals_gifs{}_{}/target.png'.format(output_dir,global_iter,ID))
        shutil.rmtree(directory)
            
def my_bar_plot(inp, err, y_lab, title, output_dir, global_iter,save=True):
    fig, ax = plt.subplots()
    ax.bar(range(1,len(inp)+1), inp,
           yerr=err,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel(y_lab, fontsize='xx-large')
    ax.set_xlabel("Z units", fontsize='xx-large')
    ax.set_xticks(range(1,len(inp)+1))
    ax.set_title(title, fontsize='xx-large')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    if save==True:
        plt.savefig('{}/{}_{}.png'.format(output_dir, title, global_iter))
    #plt.show()


def construct_z_hist(NN, loader, global_iter, output_dir, dim='depth'):
    # correct implementation of standard deviation?
    # statsitically significant or not
    # colour bars depending if gaussian or bernoulli
    # does it make sense to take difference if hybrid? gaussians pushed more together than bernoullis?
    
    with torch.no_grad():
        if dim=='depth':
            depth_z_sqd = torch.zeros(NN.z_dim_tot)
            depth_z_abs = torch.zeros(NN.z_dim_tot)
            std_sqd = torch.zeros(NN.z_dim_tot)
            std_abs = torch.zeros(NN.z_dim_tot)
            count = 0
            pbar = tqdm(total=len(loader.dataset)/loader.batch_size)
            for sample in loader:
                pbar.update(1)
                image = sample['x']
                target = sample['y']
                z_image = NN._encode(image)
                z_target = NN._encode(target)
             
                z_dist = (z_image - z_target)[:, :NN.z_dim_tot]
                depth_z_sqd += torch.mul(z_dist,z_dist).sum(0).div(z_image.size(0))
                std_sqd += torch.std(torch.mul(z_dist,z_dist),0)
                depth_z_abs += torch.abs(z_dist).sum(0).div(z_image.size(0))
                std_abs += torch.std(torch.abs(z_dist),0)
                count +=1
                
            depth_z_sqd = depth_z_sqd.div(count).numpy()
            std_sqd = std_sqd.div(count).numpy()
            depth_z_abs = depth_z_abs.div(count).numpy()
            std_abs = std_abs.div(count).numpy()
            #print(depth_z_sqd)
            
            my_bar_plot(depth_z_sqd,std_sqd/2,'Mean squared difference', 
                        'Encoding depth _1',output_dir,global_iter, save=True  )
            my_bar_plot(depth_z_abs,std_abs/2,'Mean absolute difference', 
                        'Encoding depth _2',output_dir, global_iter,save=True  )
            
    
def traverse_images():
    # what are the elements of variation & how change through them - images in video frame format?
    # make gifs out of z value bars as change image
    traversing_x 
    traversing_y  
    
    
def test_generalisation():
    # what save what images/combinations for generalisation 
    unseen_loader
            
    
def test_loss_v_occlusion():
    #test loss vs occlusion over time 
    #need to compare loss vs others to make point
    unseen_loader
    
def visualise_tsne():
    #test loss vs occlusion over time 
    #need to compare loss vs others to make point
    unseen_loader
    
    
    
def plot_decoder_img(NN, test_data, pdf_path, global_iter, sbd, type, n=20 ):
    if type =='Test':
        pdf_path = "{}/testing_recon{}.pdf".format(pdf_path, global_iter)
    elif type =='Gnrl':
        pdf_path = "{}/gnrl_recon{}.pdf".format(pdf_path, global_iter)
        
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    for i in range(n):
        sample = test_data.__getitem__(i)
        x = sample['x']
        y = sample['y']
        if sbd:
            sbd_model = spatial_broadcast_decoder()
            x = torch.unsqueeze(x, 0)
            x = sbd_model(x)
            
        x_recon = NN._decode(x)
        x = x.numpy()
        y = y.numpy()
        x_recon = F.sigmoid(x_recon).numpy()
            
        #plt.gray()    if want grey image instead of coloured 
        f, ( a1, a2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1,1]})
        #https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
        #a0.imshow(x[0,0,:,:]) #cmap='...' 
        a1.imshow(y[0,:,:])
        a2.imshow(x_recon[0,0,:,:])
        f.tight_layout()
        f.suptitle(np.around(x,2))
        pdf.savefig(f, dpi=300)
        plt.close()

    pdf.close()
    
            
def plotsave_tests(NN, test_data, pdf_path, global_iter, type, n=20, ):
    ## NN: neural network class
    ## test_data: 
    ## pdf_path: location of where to save pdf 
    ## n : number of testing images reconstruct and save
    
    if type =='Test':
        pdf_path = "{}/testing_recon{}.pdf".format(pdf_path, global_iter)
    elif type =='Gnrl':
        pdf_path = "{}/gnrl_recon{}.pdf".format(pdf_path, global_iter)
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

    for i in range(n):
        sample = test_data.__getitem__(i)
        x = sample['x']
        y = sample['y']
                
        x = torch.unsqueeze(x, 0)
        print(x.shape)
        x_recon = NN(x, train=False)
        x = x.numpy()
        y = y.numpy()
        x_recon = F.sigmoid(x_recon).numpy()
            
        #plt.gray()    if want grey image instead of coloured 
        f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1,1]})
        #https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
        a0.imshow(x[0,0,:,:]) #cmap='...' 
        a1.imshow(y[0,:,:])
        a2.imshow(x_recon[0,0,:,:])
        f.tight_layout()
        pdf.savefig(f, dpi=300)
        plt.close()

    pdf.close()
    
    

def plotLearningCurves(solver):
    """ plotting learning curves (training and testing losses and accuracies)
    """
    
    
    #fig_lc.suptitle('Learning curves \nNumber of trainable parameters: {}, \nFinal training acc: {:.2f}%, Final testing acc: {:.2f}%'.format(
     #                  solver.net.get_n_params(trainable=True), solver.gather.data['train_acc'][-1]*100,
     #                  solver.gather.data['test_acc'][-1]*100), fontsize=14)
    plt.figure(figsize = (8,8))
    plt.subplot()
    plt.plot(solver.gather.data['iter'], solver.gather.data['trainLoss'], 'r', linewidth=2.5, label = "train loss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['testLoss'], 'b', linewidth=2, label = "test loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    #plt.title("losses")
    plt.legend()
    plt.grid(True)
    plt.savefig('{}/Train_Test_loss_Curves.png'.format(solver.output_dir))
    plt.close()
    
    plt.figure(figsize = (8,8))
    plt.subplot()
    plt.plot(solver.gather.data['iter'], solver.gather.data['trainLoss'], 'coral', linewidth=2.5, label = "trainLoss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['train_recon_loss'], 'seagreen', linewidth=2.5, label = "train_recon_loss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['train_KL_loss'], 'dodgerblue', linewidth=2.5, label = "train_KL_loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    #plt.title("losses")
    plt.legend()
    plt.grid(True)
    plt.savefig('{}/Train_Loss_Curves.png'.format(solver.output_dir))
    plt.close()
    
    plt.figure(figsize = (8,8))
    plt.subplot()
    plt.plot(solver.gather.data['iter'], solver.gather.data['trainLoss'], 'r', linewidth=2.5, label = "train loss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['testLoss'], 'coral', linewidth=2.5, label = "testLoss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['test_recon_loss'], 'seagreen', linewidth=2.5, label = "train_recon_loss")
    plt.plot(solver.gather.data['iter'], solver.gather.data['test_kl_loss'], 'dodgerblue', linewidth=2.5, label = "test KL_loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    #plt.title("losses")
    plt.legend()
    plt.grid(True)
    plt.savefig('{}/Test_Loss_Curves.png'.format(solver.output_dir))
    plt.close()
    
    if solver.gather.data['grnlLoss'] :
        plt.figure(figsize = (8,8))
        plt.subplot()
        plt.plot(solver.gather.data['iter'], solver.gather.data['trainLoss'], 'r', linewidth=2.5, label = "train loss")
        plt.plot(solver.gather.data['iter'], solver.gather.data['grnlLoss'], 'coral', linewidth=2.5, label = "grnlLoss")
        plt.plot(solver.gather.data['iter'], solver.gather.data['gnrl_recon_loss'], 'seagreen', linewidth=2.5, label = "gnrl_recon_loss")
        plt.plot(solver.gather.data['iter'], solver.gather.data['gnrl_kl_loss'], 'dodgerblue', linewidth=2.5, label = "test gnrl_KL_loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        #plt.title("losses")
        plt.legend()
        plt.grid(True)
        plt.savefig('{}/Gnrl_Loss_Curves.png'.format(solver.output_dir))
    
    
    
def plotFilters(self, figIdx = None, colorLimit = 'common'):
    """ displays all filters
    colorLimit: how are colors of the filter weights scaled
        'common' = same color limit across all filters
        'individual' = each filter has its own limits
        'input' = all filters that connect to the same input (column) have the same limits
        'output' = all filters that connect to the same output (row) have the same limits

    """

    # retrieve all weights and compute min/max for possible colorLimits
    weights = [[[] for x in range(self.model.depth)] for y in range(self.model.depth)]
    actGrandMax = -np.inf
    actGrandMin = np.inf
    actRowMax = -np.inf * np.ones([self.model.depth])
    actRowMin = np.inf * np.ones([self.model.depth])
    actColMax = -np.inf * np.ones([self.model.depth])
    actColMin = np.inf * np.ones([self.model.depth])

    for ii in range(self.model.depth):
        for jj in range(self.model.depth):
            weights[ii][jj] = self.model.getWeightsByMapIndices(ii,jj)
            if len(weights[ii][jj]) > 0:
                actGrandMax = max(actGrandMax, weights[ii][jj].max())
                actGrandMin = min(actGrandMin, weights[ii][jj].min())
                actRowMax[ii] = max(actRowMax[ii], weights[ii][jj].max())
                actRowMin[ii] = min(actRowMin[ii], weights[ii][jj].min())
                actColMax[jj] = max(actColMax[jj], weights[ii][jj].max())
                actColMin[jj] = min(actColMin[jj], weights[ii][jj].min())

    # plot filters
    fig = plt.figure(figIdx)
    fig.clf()

    for ii in range(self.model.depth):
        for jj in range(self.model.depth):
            plt.subplot(self.model.depth+1, self.model.depth+1,
                        (ii+1)*(self.model.depth+1) + jj + 1)
            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            if len(weights[ii][jj]) > 0:
                if colorLimit == 'common':
                    plt.imshow(weights[ii][jj], aspect = 'auto',
                                vmin = actGrandMin, vmax = actGrandMax)
                elif colorLimit == 'individual':
                    plt.imshow(weights[ii][jj], aspect = 'auto')
                elif colorLimit == 'output':
                    plt.imshow(weights[ii][jj], aspect = 'auto',
                                  vmin = actRowMin[ii], vmax = actRowMax[ii])
                elif colorLimit == 'input':
                    plt.imshow(weights[ii][jj], aspect = 'auto',
                                  vmin = actColMin[jj], vmax = actColMax[jj])
                else:
                    raise Exception('wrong input "%s" for argument colorLimit' % colorLimit)
                plt.colorbar()

            if ii == 0:
                plt.title(self.model.mapnames[jj], fontdict={'fontsize': self.fontsize, 'fontweight': 'bold'})
            if jj == 0:
                plt.ylabel(self.model.mapnames[ii], fontdict={'fontsize': self.fontsize, 'fontweight': 'bold'})
    plt.show()
    return fig

