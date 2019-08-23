#! /bin/sh


python -W ignore main_digit_create.py \
   --n_samples_train 0 --n_samples_gnrl 10000 \
   --n_letters 2 --offset random_occluded --digit_colour_type b_w_e --linewidth 20 \
   --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/validation_border2 \
    
   
python -W ignore main_digit_create.py \
   --n_samples_train 0 --n_samples_gnrl 10000 \
   --n_letters 3 --offset random_occluded --digit_colour_type b_w_e --linewidth 20 \
   --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/validation_border3 \
    
   
    
# python -W ignore main_digit_create.py \
#      --n_samples_train 0 --n_samples_gnrl 10000  \
#      --n_letters 2 --offset random_occluded --digit_colour_type b_w_e --linewidth 20 \
#      --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/100k_2digt_BWE_2_2 \


    

#Dataset solid2: fontsize 220 w/ linewidth 5
