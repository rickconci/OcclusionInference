#! /bin/sh

#
# main submit script for Slurm on Terremoto.
#
#SBATCH --account=nklab
#SBATCH --job-name=OcclusionInference # The job name.
#SBATCH --gres=gpu:1
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=0-1:00:00 # The time the job will take to run.

 
#module load anaconda cuda92/toolkit cuda92/blas cudnn

#. activate rconci

#cd /moto/nklab/users/rc3316/OcclusionInference/Experiments/

#Command to execute Python program
python main_mod.py --train False --ckpt_name last --image_size 32 --model conv_VAE_32 \
    --z_dim 20 --n_filter 32 \
    --max_iter 200 --gather_step 10 --display_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 128 --lr 5e-4 --beta 1 \
    --output_dir /moto/nklab/users/rc3316/OcclusionInference/Experiments/Results_1 \
    --ckpt_dir /moto/nklab/users/rc3316/OcclusionInference/Experiments/Results_1 \

echo "!!" >> /moto/nklab/users/rc3316/OcclusionInference/Experiments/Results_1/LOGBOOK.txt

python main_mod.py --train False --ckpt_name last --image_size 32 --model conv_VAE_32 \
    --z_dim 20 --n_filter 32 \
    --max_iter 200 --gather_step 10 --display_step 20 \
    --dset_dir /Users/riccardoconci/Desktop/2_dig_fixed_random_bw/digts/ \
    --batch_size 128 --lr 5e-4 --beta 1 \
    --output_dir /moto/nklab/users/rc3316/OcclusionInference/Experiments/Results_2 \

echo "!!" >> /moto/nklab/users/rc3316/OcclusionInference/Experiments/Results_2/LOGBOOK.txt

#End of script