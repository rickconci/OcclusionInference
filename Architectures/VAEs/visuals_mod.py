import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision

from dataset_mod import MyDataset

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.rcParams['figure.figsize'] = [15, 15]

from dataset_mod import MyDataset
from model_mod import reparametrize



class traverse_z():
    def __init__(self, NN, example_id,  num_frames = 20):
        self.z_dim = NN.z_dim
        self.num_slice = int(1000/num_frames)
        self.num_frames = num_frames
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_test_sample = example_id['x'].to(device)
        y_test_sample = example_id['y'].to(device)
        
        #convert y_test_sample into numpy
        y_test_sample = y_test_sample.detach()
        
        
        #encode a sample image
        x_test_sample = torch.unsqueeze(x_test_sample, 0)
        z_distributions = NN._encode(x_test_sample)
        mu = z_distributions[:, :self.z_dim]
        logvar = z_distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        z_sample = z.detach()
        print(z_sample.shape)
        

        
        #create sorted normal samples & transverse_input matrix made from z encodings of sample image
        norm_samples = np.random.normal(loc=0, scale=1, size=1000)
        norm_samples.sort()
        norm_samples = torch.from_numpy(norm_samples[0::self.num_slice])
        traverse_input = torch.ones(self.num_frames*self.z_dim,1)*z_sample
        print(traverse_input.shape)
        
        #Populate matrix with individually varying Zs
        indexs = np.arange(0, self.num_frames*self.z_dim, self.z_dim)
        for i in indexs:
            z = int(i/num_frames)
            traverse_input[i:(i+self.num_frames),z] = norm_samples
            
        #create all reconstruction images
        reconst = NN._decode(traverse_input)

        #Create GIFs
        indexs = np.arange(0, self.num_frames*self.z_dim, self.z_dim)
        for i in indexs:
            #save images for each gif into the images list
            images = []
            for e in range(self.num_frames):
                #save images to make gifs into different folders
                filename = 'traversals/z{}/img{}.png'.format(int(i/self.num_frames),e)
                directory = os.path.dirname(filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torchvision.utils.save_image(F.sigmoid(reconst[i+e,0,:,:].cpu()) , filename) 
                images.append(imageio.imread(filename))
            
            #save all gifs into same folder
            filename_2 = 'traversals_gifs/traversing_z_{}.gif'.format(int(i/self.z_dim),int(i/self.z_dim))
            directory_2 = os.path.dirname(filename_2)
            if not os.path.exists(directory_2):
                    os.makedirs(directory_2)
            imageio.mimsave('traversals_gifs/traversing_z_{}.gif'.format(int(i/self.z_dim),int(i/self.z_dim)), images)
            
            #add the actual target image to the GIF image folder
            torchvision.utils.save_image(y_test_sample[0,:,:], 'traversals_gifs/target.png')
            
            
class plotsave_tests(MyDataset):
    def __init__(self, NN, test_data, pdf_path, n=20):
        ## NN: neural network class
        ## test_data: 
        ## pdf_path: location of where to save pdf 
        ## n : number of testing images reconstruct and save
        
        self.NN = NN
        self.pdf_path = "{}testing_recon.pdf".format(pdf_path)
        self.n = n
        self.test_data = test_data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.pdf_path)

        for i in range(self.n):
            sample = self.test_data.__getitem__(i)
            x = sample['x'].to(device)
            y = sample['y'].to(device)
                
            x = torch.unsqueeze(x, 0)
            x_recon, _, _ = self.NN(x)
            x = x.detach().numpy()
            y = y.detach().numpy()
            x_recon = x_recon.detach().numpy()

            f, (a0, a1, a2) = plt.subplots(r1, 3, gridspec_kw={'width_ratios': [1, 1,1]})
            a0.imshow(x[0,0,:,:])
            a1.imshow(y[0,:,:])
            a2.imshow(x_recon[0,0,:,:])
            f.tight_layout()
            pdf.savefig(f, dpi=300)
            plt.close()

        pdf.close()

    