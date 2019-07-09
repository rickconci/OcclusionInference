#! /bin/sh
  
python main_mod.py --train True --ckpt_name None --image_size 32 --model BLT_encoder \
        --z_dim 10 --n_filter 32 --encoder_target_type joint \
        --flip False --testing_method supervised_encoder \
        --max_epoch 120 --gather_step 100 --display_step 20 --save_step 200 \
        --batch_size 200 --lr 1e-4 --beta 0 \
        --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
        --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_joint_digts1 \
        --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_joint_digts1 \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_encoder_digts1/LOGBOOK.txt
python main_mod.py --train True --ckpt_name None --image_size 32 --model BLT_encoder \
        --z_dim 10 --n_filter 32 --encoder_target_type joint \
        --flip False --testing_method supervised_encoder \
                --max_epoch 120 --gather_step 100 --display_step 20 --save_step 200 \
                        --batch_size 200 --lr 5e-4 --beta 0 \
                                --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
                                        --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_joint_digts1 \
                                                --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_joint_digts1 \

EOT

#End of script

