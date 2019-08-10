#! /bin/sh


python -W ignore main_digit_create.py \
   --n_samples_train 10 --n_samples_gnrl 1 \
   --n_letters 2 --offset hidden_traverse --digit_colour_type b_w_e --linewidth 20 \
   --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/Hid_traverse_2dgt_border2 \
    
# python -W ignore main_digit_create.py \
#      --n_samples_train 0 --n_samples_gnrl 10000  \
#      --n_letters 2 --offset random_occluded --digit_colour_type b_w_e --linewidth 20 \
#      --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/100k_2digt_BWE_2_2 \


    
