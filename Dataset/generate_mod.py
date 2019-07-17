'''
Contains functions for generating stimuli
'''

import os
import random
from warnings import warn
from shutil import rmtree
import numpy as np
from PIL import Image
from character_mod import Character
from Clutter_mod import Clutter
from utils_mod import shlex_cmd, DIGITS
from io_mod import name_files
import math

import warnings
warnings.filterwarnings("ignore")


def truncated_normal_2d(minimum, maximum, mean, covariance):
    '''
    Draws a sample from a 2d truncated normal distribution
    '''
    while True:
        sample = np.random.multivariate_normal(mean, covariance, 1)
        if np.all(minimum <= sample) and np.all(sample <= maximum):
            return np.squeeze(sample)

        
def random_unoccluding_offset(rad=0.22):
    pos_1_x = np.random.uniform(-0.25, 0.25, 1)
    pos_1_y = np.random.uniform(-0.25, 0.25, 1)
    
    pos_2_x = np.random.uniform(-0.25, 0.25, 1)
    pos_2_y = np.random.uniform(-0.25, 0.25, 1)
    dist= math.sqrt((pos_1_x - pos_2_x )**2 + (pos_1_y - pos_2_y)**2)

    while dist <= 2*rad:
        print('retry')
        pos_1_x = np.random.uniform(-0.25, 0.25, 1)
        pos_1_y = np.random.uniform(-0.25, 0.25, 1)
        pos_2_x = np.random.uniform(-0.25, 0.25, 1)
        pos_2_y = np.random.uniform(-0.25, 0.25, 1)
        dist= math.sqrt((pos_1_x - pos_2_x )**2 + (pos_1_y - pos_2_y)**2)
    print('final dist', dist)
    offset_mean = [(pos_1_x,pos_1_y),(pos_2_x,pos_2_y)]
    return(offset_mean)


#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def sample_clutter(**kwargs):
    '''
    Returns a list of character objects that can be used to initialise a clutter
    object.

    kwargs:
        image_size:         as a sequence [x-size, y-size]
        n_letters:          an int for the number of characters present in each image
        font_set:           a list of TrueType fonts to be sampled from,
                            e.g. ['helvetica-bold']
        character_set:      a sequence of characters to sampled from
        face_colour_set:    a list of RGBA sequences to sample from
        edge_colour_set:    a list of RGBA sequences to sample from
        linewidth:          an int giving the width of character edges in pixels
        offset_sample_type: the distribution that offsets are drawn, 'uniform'
                            or 'gaussian'
        offset_mean:        a sequence that is the mean of the two-dimensional
                            Gaussian that the offsets are sampled from
        offset_cov:         if offset_sample_type is 'gaussian', is is the 2x2
                            covariance matrix, if offset_sample_type is
                            'uniform' then it is the parameters of the uniform
                            distribution [[x-low,x-high],[y-low,y-high]]
        size_sample_type:   the distribution that character scalings are drawn
                            from, 'gaussian' or 'truncnorm'
        size_mean:          a sequence that is the mean of the two-dimensional
                            Gaussian that the scaling coefficients are sampled from
        size_cov:           if size_sample_type is 'gaussian', is is the 2x2
                            covariance matrix, if size_sample_type is 'uniform'
                            then it is the parameters of the uniform
                            distribution [[x-low,x-high],[y-low,y-high]]
        size_min:           a sequence giving minimum scaling in each dimension
                            [x-min, y-min], only used for 'truncnorm'
        size_max:           a sequence giving minimum scaling in each dimension
                            [x-max, y-max], only used for 'truncnorm'
        fontsize:           pointsize of character as an integer

    Returns:
        clutter_sample: a list of Character objects
    '''

    image_size = kwargs.get('image_size', (512, 512))
    n_letters = kwargs.get('n_letters', 1)
    font_set = kwargs.get('font_set', ['helvetica-bold'])
    character_set = kwargs.get('character_set', DIGITS)
    digit_colour_type = kwargs.get('digit_colour_type', 'black_white')
    face_colour_set = kwargs.get('face_colour_set', [(0, 0, 0, 1.0)])
    edge_colour_set = kwargs.get('edge_colour_set', [(255, 255, 255, 1.0)])
    linewidth = kwargs.get('linewidth', 20)
    offset_sample_type = kwargs.get('offset_sample_type', 'uniform')
    offset_mean = kwargs.get('offset_mean', (0, 0.054))
    offset_cov = kwargs.get('offset_cov', ((-0.20, 0.20), (-0.12, 0.12)))
    size_sample_type = kwargs.get('size_sample_type', 'truncnorm')
    size_min = kwargs.get('size_min', (0.7, 0.7))
    size_max = kwargs.get('size_max', (1.0, 1.0))
    size_mean = kwargs.get('size_mean', (1, 1))
    size_cov = kwargs.get('size_cov', ((0, 0), (0, 0)))
    fontsize = kwargs.get('fontsize', 384)
    generalisation_set = kwargs.get('generalisation_set', False)

    g_char_set = (['3','2'],['2','3'],['9', '4'],['4','9'])

    # Sample characters without replacement
    characters = np.random.choice(character_set, n_letters, replace=False)
    
    if generalisation_set == False:
        while arreq_in_list(characters,g_char_set):
            characters = np.random.choice(character_set, n_letters,replace=False)
    elif generalisation_set == True:
        choice_indices = np.random.choice(len(g_char_set), 1, replace=True)
        characters = [g_char_set[i] for i in choice_indices]
        characters= characters[0]
        print(characters)
    
    # Initialise the clutter sample list
    clutter_sample = [None] * n_letters

    # Draw samples to get the parameters for individual characters
    char_opt = {}
    char_opt['image_size'] = image_size
    char_opt['linewidth'] = linewidth
    char_opt['fontsize'] = fontsize
    
    if offset_sample_type == 'random_unoccluded':
        offset_mean = random_unoccluding_offset(rad=0.225)

  
    for i in range(n_letters):
        char_opt['identity'] = characters[i]
        char_opt['font'] = random.choice(font_set)
        
        if n_letters ==2:
            if digit_colour_type == 'b_w':
                if i ==0:
                    char_opt['face_colour'] = random.choice(face_colour_set) 
                    char_opt['edge_colour'] = char_opt['face_colour'] 
                elif i==1:
                    face_colour = []
                    face_colour.append(char_opt['face_colour'])
                    face_colour = np.array(face_colour)
                    face_colour.flatten()
                    char_opt['face_colour'] = abs([255, 255, 255, 0] - face_colour[0]) 
                    char_opt['edge_colour'] = char_opt['face_colour']
            elif digit_colour_type == 'b_w_e':
                char_opt['face_colour'] = random.choice(face_colour_set)
                char_opt['edge_colour'] = random.choice(edge_colour_set) 
            
                
        elif n_letters >2:
            char_opt['face_colour'] = random.choice(face_colour_set) #set to black if n_letters >2
            char_opt['edge_colour'] = random.choice(edge_colour_set) #set to white if n_letters >2
        
        
        # Sample the offset
        if tuple(offset_cov) == ((0, 0), (0, 0)):
            char_opt['offset'] = offset_mean[i]

        elif offset_sample_type == 'uniform':
            x_offset = offset_mean[0] + np.random.uniform(offset_cov[0][0],
                                                       offset_cov[0][1])
            y_offset = offset_mean[1] + np.random.uniform(offset_cov[1][0],
                                                       offset_cov[1][1])
            if i <1:
                x_pos_1 = x_offset
                y_pos_1 = y_offset
           
                char_opt['offset'] = [x_offset, y_offset]

            elif i>=1:
                distance = np.sqrt((x_pos_1 - x_offset)**2 + (y_pos_1 - y_offset)**2)
                while distance <0.10 or distance >0.35:
                    #print("too close!")
                    x_offset = offset_mean[0] + np.random.uniform(offset_cov[0][0],
                                                           offset_cov[0][1])
                    y_offset = offset_mean[1] + np.random.uniform(offset_cov[1][0],
                                                               offset_cov[1][1])
                    distance = np.sqrt((x_pos_1 - x_offset)**2 + (y_pos_1 - y_offset)**2)
                char_opt['offset'] = [x_offset, y_offset]
                
                
        elif offset_sample_type == 'gaussian':
            char_opt['offset'] = np.random.multivariate_normal(offset_mean,
                                                               offset_cov)
        elif offset_sample_type == 'random_unoccluded':
            print(offset_mean[i])
            char_opt['offset'] = offset_mean[i]
        else:
            raise ValueError('{0} not a valid offset sampling type'\
            .format(offset_sample_type))
            

        # Sample the size coefficient
        if tuple(size_cov) == ((0, 0), (0, 0)):
            char_opt['size_scale'] = size_mean
        elif size_sample_type == 'gaussian':
            size_sample = np.random.multivariate_normal(size_mean, size_cov)
            char_opt['size_scale'] = (max(0, size_sample[0]), max(0, size_sample[1]))
        elif size_sample_type == 'truncated_normal_2d':
            size_sample = truncated_normal_2d(size_min, size_max, size_mean, size_cov)
            char_opt['size_scale'] = (max(0, size_sample[0]), max(0, size_sample[1]))
        else:
            raise ValueError('{0} is not a valid size sampling type'\
            .format(size_sample_type))

        clutter_sample[i] = Character(char_opt)

    return Clutter(clutter_sample)

