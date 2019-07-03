"""main.py"""

import argparse

import numpy as np

from digit_solver import Solver


def main(args):
    seed = args.seed
    np.random.seed(seed)
    data = Solver(args)
    data.create_train_test_set()
    data.create_generalisation_set()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DigitDataset')
    
    parser.add_argument('--FILENAME', default='/home/riccardo/Desktop', type=str, help='filename where to save digts folder with images')
    parser.add_argument('--seed', default=2019, type=int, help='random seed')
    parser.add_argument('--n_samples', default=10000, type=int, help='number of train/test images')
    parser.add_argument('--n_samples_gnrl', default=200, type=int, help='number of generalisation images')

    parser.add_argument('--n_letters', default=2, type=int, help='number of letters per image')
    parser.add_argument('--digit_colour_type', default='black_white', type=str, help='colours of digit face: black_white or black')
    parser.add_argument('--offset', default='fixed_unoccluded', type=str, help='type of offset of digits: fixed_unoccluded, random_unoccluded, fixed_occluded, random_occluded')
    parser.add_argument('--font_set', default='fixed', type=str, help='fixed or random font set')
    
    parser.add_argument('--image_size', nargs='+', default= (256, 256) , type=int, help='size of rendered png image')
    parser.add_argument('--fontsize', default=140, type=int, help='size of digits within image')
    parser.add_argument('--linewidth', default=7, type=int, help='width of edgde around digits')

    args = parser.parse_args()

    main(args)

