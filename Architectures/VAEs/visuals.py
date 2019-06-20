import imageio
import os
import torch
import torch.nn.functional as F
import numpy as np

class traverse_z():
    def __init__(self, NN):
        #super(traverse_z, self).__init__()
        self.z_dim = NN.z_dim
        self.NN = NN
        
        
    def create_plots(num_frames = 20):
        self.num_slice = int(1000/num_frames)
        self.num_frames = num_frames
        orig_sample = 0 #torch.randn(self.z_dim)
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

        
    def create_gif():
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