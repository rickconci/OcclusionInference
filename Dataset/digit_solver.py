"""solver_mod.py"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Image
import sys
from scipy.io import savemat
import pickle

from generate_mod import sample_clutter
import io_mod, generate_mod, Clutter_mod, utils_mod
from utils_mod import shlex_cmd, DIGITS

class Solver(object):
    def __init__(self, args):
        self.filename = args.FILENAME
        self.n_samples = args.n_samples
        self.n_samples_gnrl = args.n_samples_gnrl
        self.n_letters = args.n_letters
        self.offset = args.offset   
        self.font_set = args.font_set
        self.image_size = tuple(args.image_size)
        self.linewidth = args.linewidth
        self.fontsize = args.fontsize
        self.character_set = DIGITS
        self.digit_colour_type = args.digit_colour_type
        
        if args.digit_colour_type == "black_white" :
            self.face_colour_set = [(0, 0, 0, 1.0),(255,255,255, 1.0)]
            self.edge_colour_set = self.face_colour_set
        elif args.digit_colour_type == "black":
            self.face_colour_set = [(255,255,255, 1.0)]
            self.edge_colour_set = [(0, 0, 0, 1.0)]
            self.linewidth = 30
            self.fontsize = 140
        else:
            print("unrecognised face_colour_set option")
        
        if self.n_letters >2:
            self.face_colour_set = [(0, 0, 0, 1.0)]
            self.edge_colour_set = [(255,255,255, 1.0)]
            self.linewidth = 30
            self.fontsize = 140
     
    
        if args.offset == 'fixed_unoccluded':
            self.offset_mean = [(-0.18,-0.18),(0.18,0.18)]
            self.offset_cov = ((0,0),(0,0))
            self.offset_sample_type = 'uniform'
        elif args.offset == 'random_unoccluded':
            print("random_unoccluded!")
            self.offset_sample_type = 'random_unoccluded'
            self.offset_cov = ((-0.20, 0.20), (-0.12, 0.12))
            self.offset_mean =  ((0,0),(0,0))
        elif args.offset == 'fixed_occluded':
            self.offset_mean =  [(-0.08,-0.08),(0.08,0.08)]
            self.offset_cov = ((0,0),(0,0))
            self.offset_sample_type = 'gaussian'
        elif args.offset == 'random_occluded':
            self.offset_mean =  (0, 0.054)
            self.offset_cov = ((-0.16, 0.16), (-0.10, 0.10))
            self.offset_sample_type = 'uniform'
        else:
            raise ValueError('unrecognised offset option')
        
        if args.font_set == 'fixed':
            self.font_set = ['helvetica-bold']#['Liberation-Sans-Bold']
        elif args.font_set == 'random':
            #check if have arial-bold etc... with convert -list font 
            self.font_set = ['Liberation-Sans-Bold', 'helvetica-bold']
    
    def create_train_test_set(self):
        print("Creating train & test sets!")
        clutter_list = []
        for i in range(self.n_samples):
            clutter_list += [sample_clutter(n_letters=self.n_letters,
                                            digit_colour_type = self.digit_colour_type,
                                            face_colour_set =self.face_colour_set, 
                                            edge_colour_set= self.edge_colour_set,
                                            offset_cov = self.offset_cov,
                                            offset_mean = self.offset_mean,
                                            font_set=self.font_set,
                                            offset_sample_type = self.offset_sample_type,
                
                                            image_size = self.image_size,
                                            fontsize = self.fontsize,
                                            linewidth=self.linewidth,
                                            generalisation_set= False
                                           )]
            
        clutter_list = io_mod.name_files('{}/digts'.format(self.filename), clutter_list=clutter_list)
        io_mod.save_image_set(clutter_list, '{}/digts/digts.csv'.format(self.filename))
        
        train_size = int(self.n_samples*0.98)
        test_size = self.n_samples-train_size
        print(train_size,test_size)
        
        train_idx_to_flip = sorted(random.sample(range(0,train_size), int(train_size/2)))
        test_idx_to_flip = sorted(random.sample(range(0,test_size), int(test_size*3/4)))
        pickle.dump( train_idx_to_flip, open( "{}/digts/train_idx_to_flip.p".format(self.filename), "wb" ) )
       
        
        pbar = tqdm(total=self.n_samples)
        for i, cl in enumerate(clutter_list):
            pbar.update(1)
            if i < train_size:
                if i in train_idx_to_flip:
                    cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(self.filename,i)) 
                    cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(self.filename,i), inverse=True)
                else:
                    cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(self.filename,i))
                    cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(self.filename,i))

            elif i >= train_size:
                if i in test_idx_to_flip:
                    cl.render_occlusion(fname="{}/digts/test/orig/orig_{}".format(self.filename,i)) 
                    cl.render_occlusion(fname="{}/digts/test/inverse/inverse_{}".format(self.filename,i), inverse=True)
                else:
                    cl.render_occlusion(fname="{}/digts/test/orig/orig_{}".format(self.filename,i)) 
                    cl.render_occlusion(fname="{}/digts/test/inverse/inverse_{}".format(self.filename,i)) 

    def create_generalisation_set(self):
        print("Creating generalisation sets!")
        
        clutter_list = []
        for i in range(self.n_samples_gnrl):
            clutter_list += [sample_clutter(n_letters=self.n_letters,
                                            digit_colour_type = self.digit_colour_type,
                                            face_colour_set =self.face_colour_set, 
                                            edge_colour_set= self.edge_colour_set,
                                            offset_cov = self.offset_cov,
                                            offset_mean = self.offset_mean,
                                            font_set=self.font_set,
                                            offset_sample_type = self.offset_sample_type,
                
                                            image_size = self.image_size,
                                            fontsize = self.fontsize,
                                            linewidth=self.linewidth,
                                            generalisation_set= True
                                           )]
            
        #clutter_list = io_mod.name_files('{}/digts'.format(self.filename), clutter_list=clutter_list)
        io_mod.save_image_set(clutter_list, '{}/digts/digts_gnrl.csv'.format(self.filename))
        
        gnrl_idx_to_flip = random.sample(range(0,self.n_samples_gnrl), int(self.n_samples_gnrl*3/4))

        pbar = tqdm(total=self.n_samples_gnrl)
        for i, cl in enumerate(clutter_list):
            pbar.update(1)
            if i in gnrl_idx_to_flip:
                cl.render_occlusion(fname="{}/digts/gnrl/orig/orig_{}".format(self.filename,i)) 
                cl.render_occlusion(fname="{}/digts/gnrl/inverse/inverse_{}".format(self.filename,i), inverse=True)
            else:
                cl.render_occlusion(fname="{}/digts/gnrl/orig/orig_{}".format(self.filename,i)) 
                cl.render_occlusion(fname="{}/digts/gnrl/inverse/inverse_{}".format(self.filename,i), inverse=False)
            
