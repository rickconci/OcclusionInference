#! /bin/sh


python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 5e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/B_2  \

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 35 --n_rep 4 --kernel_size 6 --padding 2 \
    --optim_type Adam --batch_size 100 --lr 5e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/B_matched_2  \


python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BLT --decoder BLT --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 5e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BLT_2  \

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BL --decoder BLT --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BL_lr0_001_2  \


python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BT --decoder BLT --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 5e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_3digt_BWE/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/3_digts/arch/BT_2  \


#End of script

