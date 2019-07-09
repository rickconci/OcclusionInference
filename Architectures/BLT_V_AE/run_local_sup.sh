#! /bin/sh

python main_mod.py --train True --ckpt_name None --image_size 32 --model BLT_encoder \
    --z_dim 10 --n_filter 32 --encoder_target_type joint \
    --flip False --testing_method supervised_encoder \
    --max_epoch 10 --gather_step 100 --display_step 5 --save_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 200 --lr 5e-4 --beta 0 \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \

cat <<EOT>> /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1/LOGBOOK.txt
python main_mod.py --train True --ckpt_name None --image_size 32 --model BLT_encoder  \
    --testing_method supervised_encoder \
    --z_dim 10 --n_filter 32 --encoder_target_type joint --flip False \
    --max_epoch 10 --gather_step 100 --display_step 5 --save_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 200 --lr 5e-4 --beta 0 \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \

EOT

#End of script

