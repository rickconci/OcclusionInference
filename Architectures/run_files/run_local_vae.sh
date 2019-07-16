#! /bin/sh

python main_mod.py --train False --ckpt_name None --testing_method unsupervised \
    --model BLT_hybrid_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 0.17 --gather_step 5 --display_step 10 --save_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae \
    --ckpt_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae  \
