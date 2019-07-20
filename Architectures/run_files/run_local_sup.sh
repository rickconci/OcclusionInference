#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder BLT --sbd True --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type SGD --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 10 --gather_step 50 --display_step 5 --save_step 20 \
     --dset_dir /Users/riccardoconci/Desktop/100k_bw_ro/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_sup \
    --ckpt_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_vsup  \

#End of script

