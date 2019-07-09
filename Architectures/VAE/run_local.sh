#! /bin/sh

python main_mod.py --train False --ckpt_name last --image_size 32 --model hybrid_VAE \
    --z_dim 20 --n_filter 32 \
    --max_epoch 10 --gather_step 100 --display_step 5 --save_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 200 --lr 5e-4 --beta 0 \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_hyb_VAE_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_hyb_VAE_digt1 \

cat <<EOT>> /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_hyb_VAE_digt1/LOGBOOK.txt
python main_mod.py --train True --ckpt_name None --image_size 32 --model hybrid_VAE \
    --z_dim 20 --n_filter 32 \
    --max_epoch 60 --gather_step 100 --display_step 5 --save_step 100 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 200 --lr 5e-4 --beta 0 \
    --output_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_hyb_VAE_digt1 \
    --ckpt_dir /Users/riccardoconci/Desktop/code/Zuckerman_Project/results/Results_hyb_VAE_digt1 \
EOT

#End of script