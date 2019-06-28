#! /bin/sh

python main_mod.py --train True --ckpt_name None --image_size 32 --model conv_AE \
    --z_dim 20 --n_filter 32 \
    --max_iter 6000 --gather_step 100 --display_step 20 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
    --batch_size 128 --lr 5e-4 --beta 0 \
    --output_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1 \

fc -ln -10 >> /home/riccardo/Desktop/Experiments/Results_AE_digts1/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --image_size 32 --model conv_AE \
    --z_dim 10 --n_filter 32 \
    --max_iter 6000 --gather_step 100 --display_step 20 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
    --batch_size 128 --lr 5e-4 --beta 0 \
    --output_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1_z10 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1_z10 \

fc -ln -10 >> /home/riccardo/Desktop/Experiments/Results_AE_digts1_z10/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --image_size 32 --model conv_AE \
    --z_dim 10 --n_filter 64 \
    --max_iter 6000 --gather_step 100 --display_step 20 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_1/digts/ \
    --batch_size 128 --lr 5e-4 --beta 0 \
    --output_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1_nfilt64 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_AE_digts1_nfilt64 \

fc -ln -10 >> /home/riccardo/Desktop/Experiments/Results_AE_digts1_nfilt64/LOGBOOK.txt

#End of script