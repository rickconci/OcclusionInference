#! /bin/sh

python -W ignore main_digit_create.py --n_letters 2 --n_samples 20 --n_samples_gnrl 10  \
    --digit_colour_type black_white --offset random_unoccluded  --font_set fixed \
    --FILENAME /home/riccardo/Desktop/Digts_Exp_5 \
    
python -W ignore main_digit_create.py --n_letters 2 --n_samples 10000 --n_samples_gnrl 200  \
    --digit_colour_type black_white --offset random_occluded  --font_set fixed \
    --FILENAME /home/riccardo/Desktop/Digts_Exp_6 \