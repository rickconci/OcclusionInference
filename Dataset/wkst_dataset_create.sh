#! /bin/sh

python -W ignore main_digit_create.py --n_samples 10000 --n_letters 2 --digit_colour_type black_white \
    --offset random_occluded  --font_set fixed \
    --FILENAME /home/riccardo/Desktop/Digts_Exp_4 \
    
python -W ignore main_digit_create.py --n_samples 10000 --n_letters 2 --digit_colour_type black_white \
    --offset random_unoccluded  --font_set fixed \
    --FILENAME /home/riccardo/Desktop/Digts_Exp_3 \