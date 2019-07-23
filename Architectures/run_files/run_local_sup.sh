#! /bin/sh

python main_mod.py --train True --ckpt_name None --testing_method supervised_encoder \
    --encoder BT --decoder BLT --sbd False --encoder_target_type depth_black_white_xy_xy --n_filter 32 --n_rep 4 \
    --optim_type Adam --batch_size 100 --lr 1e-3 --beta 0 \
    --max_epoch 0.2 --gather_step 50 --display_step 5 --save_step 200 \
     --dset_dir /Users/riccardoconci/Desktop/100k_bw_ro/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_loc_sup \

#End of script

