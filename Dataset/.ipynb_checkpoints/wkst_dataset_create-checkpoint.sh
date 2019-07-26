#! /bin/sh

python -W ignore main_digit_create.py \
    --n_samples_train 100000 --n_samples_gnrl 10000  \
    --n_letters 3 --offset random_occluded --digit_colour_type b_w_e --linewidth 25 \
    --fontsize 160 --FILENAME /home/riccardo/Desktop/Data/100k_3digt_bwe \
    
python -W ignore main_digit_create.py \
    --n_samples_train 100000 --n_samples_gnrl 10000  \
    --n_letters 2 --offset random_occluded --digit_colour_type b_w_e --linewidth 25 \
    --fontsize 160 --FILENAME /home/riccardo/Desktop/Data/100k_2digt_bwe \