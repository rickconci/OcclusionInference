#! /bin/sh

python main_mod.py --train True --image_size 32 --model conv_VAE_32 --z_dim 20 \
    --max_iter 400 --gather_step 10 --display_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 128 --lr 5e-4 --beta 1 \

