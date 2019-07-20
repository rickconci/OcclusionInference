#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcast_decoder False \
     --max_epoch 100 --gather_step 500 --display_step 200 --save_step 10000 \
    --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
   --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 100 --gather_step 500 --display_step 200 --save_step 10000 \

EOT
