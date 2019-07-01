#! /bin/sh

python main_mod.py --train True --ckpt_name None --image_size 32 --model conv_AE \
    --z_dim 20 --n_filter 32 \
    --max_iter 6000 --gather_step 100 --display_step 20 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_2/digts/ \
    --batch_size 128 --lr 5e-4 --l1_penalty 5e-1 \
    --output_dir /home/riccardo/Desktop/Experiments/Results_AE_05l1_digts2 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_AE_05l1_digts2 \

history >> /home/riccardo/Desktop/Experiments/Results_AE_05l1_digts2/LOGBOOK.txt

#End of script