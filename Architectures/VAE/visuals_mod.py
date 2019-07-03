import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision
import shutil
from tqdm import tqdm

from dataset_mod import MyDataset

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.rcParams['figure.figsize'] = [15, 15]

from dataset_mod import MyDataset
from model_mod import reparametrize_gaussian, reparametrize_bernoulli



def traverse_z(NN, example_id, ID, output_dir, global_iter, model ,num_frames = 100 ):
    z_dim = NN.z_dim_tot
    z_dim_gauss = NN.z_dim_gauss
    z_dim_bern = NN.z_dim_bern
    
    x_test_sample = example_id['x']
    y_test_sample = example_id['y']
    x_test_sample = torch.unsqueeze(x_test_sample, 0)
    
    #encode a sample image
    z_distributions = NN._encode(x_test_sample)
    if model == 'gauss_VAE':
        z_sample = z_distributions[:, :z_dim_gauss]
    elif model == 'bern_VAE':
        z_sample = z_distributions[:, :z_dim_bern]
    elif model == 'hybrid_VAE':
        p = z_distributions[:, :z_dim_bern]
        mu = z_distributions[:, z_dim_bern:z_dim_gauss]
        z_sample = torch.cat((p, mu),1)
   
    x_recon = NN._decode(z_sample)
   
    num_slice = int(1000/num_frames)

    if model == 'gauss_VAE':
        #create sorted normal samples & transverse_input matrix made from z encodings of sample image
        dist_samples = np.random.normal(loc=0, scale=1, size=1000)
        dist_samples.sort()
        dist_samples = torch.from_numpy(dist_samples[0::num_slice])
    elif model == 'bern_VAE':
        dist_samples = np.random.uniform(low=0, high=1, size=1000)
        dist_samples.sort()
        dist_samples = torch.from_numpy(dist_samples[0::num_slice])
    #elif model =='hybrid_VAE':
        
            
    traverse_input = torch.mul(torch.ones(num_frames*z_dim,1),z_sample)

    #print(traverse_input.shape)

    #Populate matrix with individually varying Zs
    indexs = np.arange(0, num_frames*z_dim, num_frames)
    for i in indexs:
        z = int(i/num_frames)
        traverse_input[i:(i+num_frames),z] = dist_samples

    #create all reconstruction images
    reconst = NN._decode(traverse_input)

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
    plt.show()


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
        x_recon = NN(x, train=False)
        x = x.detach().numpy()
        y = y.detach().numpy()
        x_recon = F.sigmoid(x_recon).detach().numpy()
            
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

    