#! /bin/sh

python main_mod.py --train True --ckpt_name last --image_size 32 --model conv_VAE_32 \
    --z_dim 20 --n_filter 32 \
    --max_iter 4800 --gather_step 10 --display_step 20 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
    --batch_size 128 --lr 5e-4 --beta 1 \

