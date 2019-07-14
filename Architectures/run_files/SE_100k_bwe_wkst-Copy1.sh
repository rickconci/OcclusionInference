#! /bin/sh

python main_mod.py --train True --ckpt_name None --model BLT_orig \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_orig_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_orig_joint_100k_bwe_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_orig_joint_100k_bwe_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --model FF \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_orig_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_orig_joint_100k_bwe_ro \

EOT

#End of script


python main_mod.py --train True --ckpt_name None --model BLT_mod \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_mod_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_mod_joint_100k_bwe_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_mod_joint_100k_bwe_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --model BLT_mod \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_mod_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_mod_joint_100k_bwe_ro \

EOT

#End of script


python main_mod.py --train True --ckpt_name None --model FF \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_joint_100k_bwe_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_FF_joint_100k_bwe_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --model FF \
    --testing_method supervised_encoder --encoder_target_type joint --z_dim 10 \
    --batch_size 100 --lr 1e-2 --beta 0 \
    --max_epoch 100 --gather_step 40 --display_step 40 --save_step 200 \
    --dset_dir /home/riccardo/Desktop/100k_bwe_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_FF_joint_100k_bwe_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_FF_joint_100k_bwe_ro \

EOT

#End of script