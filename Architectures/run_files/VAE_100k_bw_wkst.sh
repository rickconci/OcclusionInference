#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder B --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE/hybrid/B_B_21b_5g \

cat <<EOT>> /home/riccardo/Desktop/Experiments/VAE/hybrid/B_B_21b_5g/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder B --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BL --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE/hybrid/BL_B_21b_5g \

cat <<EOT>> /home/riccardo/Desktop/Experiments/VAE/hybrid/BL_B_21b_5g/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BL --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \

EOT

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BT --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE/hybrid/BT_B_21b_5g \

cat <<EOT>> /home/riccardo/Desktop/Experiments/VAE/hybrid/BT_B_21b_5g/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BL --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BLT --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/VAE/hybrid/BLT_B_21b_5g \

cat <<EOT>> /home/riccardo/Desktop/Experiments/VAE/hybrid/BLT_B_21b_5g/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --encoder BLT --decoder B --sbd False --z_dim_bern 21 --z_dim_gauss 5 --n_filter 32 \
    --optim_type Adam  --lr 1e-3 --beta 1 --gamma 1 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 50 --save_step 10000 \

EOT