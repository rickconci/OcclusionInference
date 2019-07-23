#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/arch/B  \

cat <<EOT>> /home/riccardo/Desktop/Experiments/encoder_sup/arch/B/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BL --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/arch/BL \

cat <<EOT>> /home/riccardo/Desktop/Experiments/encoder_sup/arch/BL/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BL --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \

EOT

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BT --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/arch/BT \

cat <<EOT>> /home/riccardo/Desktop/Experiments/encoder_sup/arch/BT/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BT --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \

EOT

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BLT --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/encoder_sup/arch/BLT \

cat <<EOT>> /home/riccardo/Desktop/Experiments/encoder_sup/arch/BLT/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BLT --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
EOT




#End of script


