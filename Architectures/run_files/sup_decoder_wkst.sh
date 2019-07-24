#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 36 --n_rep 4 --kernel_size 6 --padding 2 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/B_matched  \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/B_matched/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 36 --n_rep 4 --kernel_size 6 --padding 2 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/B  \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/B/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \

EOT

#End of script


