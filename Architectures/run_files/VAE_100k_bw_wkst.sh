#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_gauss_150k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_100k_bw_ro \

EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro \

EOT



python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_brnl_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 0 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_brnl_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_brnl_100k_bw_ro \
    
cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_brnl_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 0 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_brnl_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_brnl_100k_bw_ro \

EOT

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_hybrid_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_hyb_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_hyb_100k_bw_ro \
    
cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_hybrid_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_hyb_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_hyb_100k_bw_ro \
EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_brnl_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_brnl_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_brnl_100k_bw_ro \
    
cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_brnl_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_brnl_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_brnl_100k_bw_ro \
EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_hybrid_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_hybrid_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_hybrid_100k_bw_ro \
    
cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model FF_hybrid_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_hybrid_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_hybrid_100k_bw_ro \
EOT


python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_beta3_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_beta3_100k_bw_ro \
    
cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_gauss_beta3_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 3 --gamma 1 \
    --optim_type Adam --spatial_broadcaster False \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/150k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_beta3_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_beta3_100k_bw_ro \
EOT