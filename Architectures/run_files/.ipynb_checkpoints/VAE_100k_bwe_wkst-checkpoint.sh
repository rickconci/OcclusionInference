#! /bin/sh

    

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder B --decoder BLT --sbd False --z_dim_bern 0 --z_dim_gauss 24 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE_2/BLT_BLT_g25_lr_0_001 \
    
    
    
python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BLT --decoder BLT --sbd False --z_dim_bern 0 --z_dim_gauss 24 --n_filter 32 \
    --optim_type Adam  --lr 5e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE_2/BLT_BLT_g25_lr_0_005 \
    