#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/B  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/B   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/B/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BL --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BL  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BL   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/BL/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BL --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BT  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BT   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/BT/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BTL  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/BTL   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/BTL/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd True --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/SBD  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/arch/SBD   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/arch/SBD/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd True --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 64 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/n_filt/BLT_64  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/n_filt/BLT_64   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/n_filt/BLT_64/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 64 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT



python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 8 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \
     --dset_dir /home/riccardo/Desktop/better_100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/n_rep/BLT_nrep_8  \
    --ckpt_dir /home/riccardo/Desktop/Experiments/decoder_sup/n_rep/BLT_nrep_8   \


cat <<EOT>> /home/riccardo/Desktop/Experiments/decoder_sup/n_rep/BLT_nrep_8/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 8 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 50 --gather_step 200 --display_step 50 --save_step 1000 \

EOT


#End of script


