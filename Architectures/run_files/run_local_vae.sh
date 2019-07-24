#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BLT --decoder BLT --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 0.1 --gather_step 20 --display_step 5 --save_step 60 \
    --dset_dir /Users/riccardoconci/Desktop/100k_bw_ro/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vae \
