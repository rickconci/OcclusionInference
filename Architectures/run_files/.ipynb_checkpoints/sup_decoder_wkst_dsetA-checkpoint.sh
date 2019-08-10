#! /bin/sh



python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bw/col_binding/B_lr_0_001  \


# python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
#     --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
#     --n_filter 35 --n_rep 4 --kernel_size 6 --padding 2 \
#     --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
#     --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
#      --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
#     --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bw/col_binding/B_matched_lr_0_001  \


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bw/col_binding/BLT_lr_0_001  \


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BL --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bw/col_binding/BL_lr_0_001  \


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BT --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2_digt_BW/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bw/col_binding/BT_lr_0_001  \




#End of script


