#! /bin/sh

python main_mod.py --train True --ckpt_name None --model BLT_orig \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 1 --gather_step 5 --display_step 6 --save_step 30 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_BLT_sup_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/ZuckermanProject/results/Results_BLT_sup_digt1 \

#End of script
