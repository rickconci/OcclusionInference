#! /bin/sh

python main.py --dataset digits --model mmd --max_epoch 250 --batch_size 100 \
    --lr 1e-3 --beta1 0.5 --beta2 0.999 --z_dim 20 --z_var 2 --reg_weight 100 \
    --viz_name wae_mmd_digits --dset_dir /Users/riccardoconci/Desktop/100k_bw_ro/digts/ \
#/home/riccardo/Desktop/better_100k_bw_ro/digts/ 