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
        #Image Params
        self.n_letters = args.n_letters
        self.offset = args.offset
        self.digit_colour_type = args.digit_colour_type
        self.font_set = args.font_set
        #image hyperparams
        self.fontsize = args.fontsize
        self.linewidth = args.linewidth
        self.image_size = tuple(args.image_size)
        self.character_set = DIGITS
        #dataset hyperparams
        self.filename = args.FILENAME
        self.n_samples_tot = args.n_samples_tot
        self.n_samples_test = args.n_samples_test
        self.n_samples_gnrl = args.n_samples_gnrl
        self.unflip = args.unflip
        
        
        if self.unflip:
            print("UNFLIP IS ON: HALF OF IMAGES WILL NOT BE FLIPPED")
            
      
        
        #Image params
        
        if args.digit_colour_type == "b_w" :
            self.face_colour_set = [(0, 0, 0, 1.0),(255,255,255, 1.0)]
            self.edge_colour_set = self.face_colour_set
        elif args.digit_colour_type == "b_w_e":
            self.face_colour_set = [(255,255,255, 1.0)]
            self.edge_colour_set = [(0, 0, 0, 1.0)]
            self.linewidth = 25
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
            self.double_input = raw_var("Double_input? (True or False):")
            if self.double_input:
                print("DOUBLE INPUT IS ON:both original and flipped in input")
        else:
            raise ValueError('unrecognised offset option')
        
        if args.font_set == 'fixed':
            self.font_set = ['Liberation-Sans-Bold']#['helvetica-bold']#
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
        
        train_size = int(self.n_samples_tot-self.n_samples_test)
        test_size = self.n_samples_test
        print(train_size,test_size)
        
        
        if self.unflip:
            train_idx_to_flip = sorted(random.sample(range(0,train_size), int(train_size/2)))
            test_idx_to_flip = sorted(random.sample(range(0,test_size), int(test_size*3/4)))
            pickle.dump( train_idx_to_flip, open( "{}/digts/train_idx_to_flip.p".format(
                self.filename), "wb" ) )
       
        
        pbar = tqdm(total=self.n_samples_tot)
        for i, cl in enumerate(clutter_list):
            pbar.update(1)
            if i < train_size:
                
                if self.unflip:
                    if i in train_idx_to_flip:
                        cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(
                            self.filename,i)) 
                        cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(
                            self.filename,i), inverse=True)
                    else:
                        cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(
                            self.filename,i))
                        cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(
                            self.filename,i))
                
                elif self.double_input:
                    cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(
                        self.filename,i))
                    cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(
                        self.filename,i+train_size), inverse=True)
                    
                    cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(
                        self.filename,i), inverse=True)
                    cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(
                        self.filename,i+train_size))
                    
                else:
                    cl.render_occlusion(fname="{}/digts/train/orig/orig_{}".format(
                        self.filename,i))
                    cl.render_occlusion(fname="{}/digts/train/inverse/inverse_{}".format(
                        self.filename,i), inverse=True)
                    

            elif i >= train_size:
                cl.render_occlusion(fname="{}/digts/test/orig/orig_{}".format(
                        self.filename,i))
                cl.render_occlusion(fname="{}/digts/test/inverse/inverse_{}".format(
                        self.filename,i), inverse=True)

                    
                    
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
        
        pbar = tqdm(total=self.n_samples_gnrl)
        for i, cl in enumerate(clutter_list):
            pbar.update(1)
            cl.render_occlusion(fname="{}/digts/gnrl/orig/orig_{}".format(self.filename,i)) 
            cl.render_occlusion(fname="{}/digts/gnrl/inverse/inverse_{}".format(
                self.filename,i), inverse=True)
           
            
