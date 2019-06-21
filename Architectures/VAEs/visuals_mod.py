import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
plt.rcParams['figure.figsize'] = [15, 15]

from dataset_mod import MyDataset



class traverse_z():
    def __init__(self, NN, example_input, num_frames = 20):
        #super(traverse_z, self).__init__()
        self.z_dim = NN.z_dim
        self.NN = NN
        
        self.num_slice = int(1000/num_frames)
        self.num_frames = num_frames
        orig_sample = 0 #torch.randn(self.z_dim)
        
        orig_sample = self.NN._encode(example_input)
        
        norm_samples = np.random.normal(loc=0, scale=1, size=1000)
        norm_samples.sort()
        norm_samples = torch.from_numpy(norm_samples[0::self.num_slice])
        traverse_input = torch.ones(self.num_frames*self.z_dim,1)*orig_sample
        indexs = np.arange(0, self.num_frames*self.z_dim, self.z_dim)
        
        for i in indexs:
            z = int(i/num_frames)
            print(i, z)
            traverse_input[i:(i+self.num_frames),z] = norm_samples
            
        reconst = self.NN._decode(traverse_input)

        indexs = np.arange(0, self.num_frames*self.z_dim, self.z_dim)
        for i in indexs:
            images = []
            for e in range(self.num_frames):
                #print(i+e)
                filename = 'traversals/z{}/img{}.png'.format(int(i/self.num_frames),e)
                directory = os.path.dirname(filename)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torchvision.utils.save_image(F.sigmoid(reconst[i+e,0,:,:].cpu()) , filename) #remove the F.sigmoid if already at end of NN

                images.append(imageio.imread(filename))
            filename_2 = 'traversals_gifs/traversing_z_{}.gif'.format(int(i/self.z_dim),int(i/self.z_dim))
            directory_2 = os.path.dirname(filename_2)
            if not os.path.exists(directory_2):
                    os.makedirs(directory_2)
            imageio.mimsave('traversals_gifs/traversing_z_{}.gif'.format(int(i/self.z_dim),int(i/self.z_dim)), images)
            
            
            
class plotsave_tests():
    def __init__(self, NN, MyDataset, test_image_paths, test_target_paths, pdf_path, n=20):
        ## NN: neural network class
        ## MyDataset: Image loader class
        ## test_image_paths: location of testing x image
        ## test_target_paths: location of testing y image
        ## pdf_path: location of where to save pdf 
        ## n : number of testing images reconstruct and save
        
        self.NN = NN
        self.MyDataset = MyDataset
        self.test_image_paths = test_image_paths
        self.test_target_paths = test_target_paths
        self.pdf_path = "{}testing_recon.pdf".format(pdf_path)
        self.n = n
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dset = self.MyDataset
        test_data = dset(self.test_image_paths,self.test_target_paths, image_size= 32)

        pdf = matplotlib.backends.backend_pdf.PdfPages(self.pdf_path)
        

        for i in range(self.n):
            sample = test_data.__getitem__(i)
            x = sample['x'].to(device)
            y = sample['y'].to(device)
                
            x = torch.unsqueeze(x, 0)
            x_recon, _, _ = self.NN(x)
            x = x.detach().numpy()
            y = y.detach().numpy()
            x_recon = x_recon.detach().numpy()

            f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1,1]})
            a0.imshow(x[0,0,:,:])
            a1.imshow(y[0,:,:])
            a2.imshow(x_recon[0,0,:,:])
            f.tight_layout()
            pdf.savefig(f, dpi=300)
            plt.close()

        pdf.close()

    