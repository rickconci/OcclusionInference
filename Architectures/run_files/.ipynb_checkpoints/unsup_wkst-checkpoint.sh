#! /bin/sh

# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 6 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim6_2 \

# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 12 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim12_2 \


# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 18 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim18_2 \
    
    
# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 24 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim24_2 \
    
    
# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 30 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim30_2 \


# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 6 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim6 \
    
    
# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 12 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim12 \
    

# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 18 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim18 \



python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
    --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 30 --z_dim_gauss 0  \
    --optim_type Adam  --lr 1e-3 --batch_size 100 \
    --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
    --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim30 \



# python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
#     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 36 --z_dim_gauss 0  \
#     --optim_type Adam  --lr 1e-3 --batch_size 100 \
#     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
#     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/3_digts/BLT_BLT_depth_zdim36 \


#End of script
