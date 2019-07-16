#! /bin/sh

#
# main submit script for Slurm on Terremoto.
#
#SBATCH --account=nklab
#SBATCH --job-name=OcclusionInference # The job name.
#SBATCH --gres=gpu:1
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=0-6:00:00 # The time the job will take to run.

 
module load anaconda cuda92/toolkit cuda92/blas cudnn

. activate rconci

cd /moto/nklab/users/rc3316/OcclusionInference/Architectures/run_files

#Command to execute Python program
python main_mod.py --train True --ckpt_name last --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster True \
     --max_epoch 100 --gather_step 200 --display_step 50 --save_step 10000 \
    --dset_dir ~/moto/nklab/users/rc3316/100k_bw_ro/digts/ \
    --output_dir ~/moto/nklab/users/rc3316/Experiments/Results_BLT_gauss_SB_100k_bw_ro \
    --ckpt_dir ~/moto/nklab/users/rc3316//Experiments/Results_BLT_gauss_SB_100k_bw_ro \

cat <<EOT>> /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SB_100k_bw_ro/LOGBOOK.txt

python main_mod.py --train True --ckpt_name None --testing_method unsupervised \
    --model BLT_gauss_VAE --z_dim 20 --batch_size 100 --lr 5e-4 --beta 1 --gamma 1 \
    --optim_type Adam --spatial_broadcaster True \
     --max_epoch 30 --gather_step 100 --display_step 50 --save_step 1000 \
    --dset_dir /home/riccardo/Desktop/100k_bw_ro/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SB_100k_bw_ro \
    --ckpt_dir /home/riccardo/Desktop/Experiments/Results_BLT_gauss_SB_100k_bw_ro \

EOT


#End of script