#! /bin/sh

python main_mod.py --train True --ckpt_name None --image_size 32 --model conv_AE_mini \
    --z_dim 20 --n_filter 10 \
    --max_iter 400 --gather_step 100 --display_step 20 --save_step 200\
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 128 --lr 5e-4 --l1_penalty 5e-1 \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_AE_mini_05l1_digts1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_AE_mini_05l1_digts1 \

fc -ln -10 >> /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_AE_mini_05l1_digts1/LOGBOOK.txt

#End of script