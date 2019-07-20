#! /bin/sh

python main_mod.py --train False --ckpt_name None --testing_method unsupervised \
    --encoder B --decoder B --sbd False --z_dim_bern 20 --z_dim_gauss 0 --n_filter 32 \
    --optim_type Adam  --lr 5e-4 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 1 --gather_step 500 --display_step 5 --save_step 500 \
    --dset_dir /Users/riccardoconci/Desktop/100k_bw_ro/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae \
    --ckpt_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae  \
