import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision
import shutil

from dataset_mod import MyDataset

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.rcParams['figure.figsize'] = [15, 15]

from dataset_mod import MyDataset



def traverse_z(NN, example_id, ID, output_dir, global_iter, model ,num_frames = 100 ):
    z_dim = NN.z_dim
    num_slice = int(1000/num_frames)

    x_test_sample = example_id['x']
    y_test_sample = example_id['y']

    #convert y_test_sample into numpy
    y_test_sample = y_test_sample.detach()


    #encode a sample image
    x_test_sample = torch.unsqueeze(x_test_sample, 0)
    z_distributions = NN._encode(x_test_sample)
    mu = z_distributions[:, :z_dim]
    z_sample = mu.detach()
    x_recon = NN._decode(z_sample)
    #print(z_sample.shape)
    print(z_sample)
        

    if model == 'conv_AE':
        dist_samples = np.random.uniform(low=-35, high=35, size=1000)
        dist_samples.sort()
        dist_samples = torch.from_numpy(dist_samples[0::num_slice])
            
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
            
            
def plotsave_tests(NN, test_data, pdf_path, global_iter, n=20):
    ## NN: neural network class
    ## test_data: 
    ## pdf_path: location of where to save pdf 
    ## n : number of testing images reconstruct and save
        
    pdf_path = "{}/testing_recon{}.pdf".format(pdf_path, global_iter)
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

    