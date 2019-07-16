#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcast_decoder True \
     --max_epoch 1 --gather_step 50 --display_step 5 --save_step 100 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae \
    --ckpt_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae  \
