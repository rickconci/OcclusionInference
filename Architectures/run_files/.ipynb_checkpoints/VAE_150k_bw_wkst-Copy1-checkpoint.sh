#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro/LOGBOOK.txt

python main_mod.py --train False --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro \

EOT

#End of script

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_150k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_150k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_150k_bw_ro \


EOT

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 1e-2 --beta 1 --gamma 1 \
    --optim_type SGD --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SGD_lr001_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SGD_lr001_150k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SGD_lr001_150k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 1e-2 --beta 1 --gamma 1 \
    --optim_type SGD --spatial_broadcaster False \
     --max_epoch 100 --gather_step 20 --display_step 20 --save_step 500 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SGD_lr001_150k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SGD_lr001_150k_bw_ro \

EOT