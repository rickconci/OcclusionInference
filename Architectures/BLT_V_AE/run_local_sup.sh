#! /bin/sh

python main_mod.py --train True --ckpt_name None --model FF \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 5e-4 --beta 0 \
    --max_epoch 1 --gather_step 5 --display_step 5 --save_step 10 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \

cat <<EOT>> /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1/LOGBOOK.txt
\n
python main_mod.py --train True --ckpt_name None --model FF \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 5e-4 --beta 0 \
    --max_epoch 1 --gather_step 5 --display_step 5 --save_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_BLT_sup_digt1 \

EOT

#End of script

