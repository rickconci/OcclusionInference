"""solver_mod.py"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Image


class Solver(object):
    def __init__(self, args):
        self.n_samples = args.n_samples
        self.n_letters = args.n_letters
        self.linewidth = args.linewidth
        self.image_size = args.image_size
        self.font_set = args.font_set
        self.character_set = args.character_set
        self.face_colour_set = args.face_colour_set
        self.edge_colour_set = args.edge_colour_set
        self.linewidth = args.linewidth
        self.offset_sample_type = args.offset_sample_type
        self.offset_mean = args.offset_mean
        self.offset_cov = args.offset_cov
        self.size_sample_type = args.size_sample_type
        self.size_min = args.size_min
        self.size_max = args.size_max
        self.size_mean = args.size_mean
        self.size_cov = args.size_cov
        self.fontsize = args.fontsize
        
        clutter_list = []
        for i in range(n_samples):
            clutter_list += [sample_clutter(font_set=self.font_set,
                                            n_letters=self.n_letters,
                                            linewidth=self.linewidth, 
                                            image_size = self.image_size
                                            face_colour_set =self.face_colour_set, 
                                            edge_colour_set= self.edge_colour_set,
                                            offset_cov = self.offset_cov,
                                            offset_mean = self.offset_mean 
                                            character_set = self.character_set
                                           
                                           )]