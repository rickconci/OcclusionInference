#! /bin/sh

python main_mod.py --train True --ckpt_name None --image_size 32 --model hybrid_VAE \
    --z_dim 32 --n_filter 32 --batch_size 200 --lr 5e-4 --beta 1 --gamma 1 \
    --max_epoch 60 --gather_step 100 --display_step 20 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_5/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_hyb_VAE_digts5 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_hyb_VAE_digts5 \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_hyb_VAE_digts5/LOGBOOK.txt
python main_mod.py --train True --ckpt_name None --image_size 32 --model hybrid_VAE \
    --z_dim 32 --n_filter 32 --batch_size 200 --lr 5e-4 --beta 1 --gamma 1 \
    --max_epoch 60 --gather_step 100 --display_step 20 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/Digts_Exp_5/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_hyb_VAE_digts5 \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_hyb_VAE_digts5 \
EOT

#End of script