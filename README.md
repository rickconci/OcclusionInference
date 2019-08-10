# OcclusionInference

## Background

This repository contains the code for the my MPhil Thesis in Computational Biology submitted the department of 
Applied Mathematics and Theretical Physics, University of Cambidge, in August 2019. 


In this thesis I test the hypothesis I investigate the role of recurrent connections in inference and graphics by building
different artificial neural network models, and training and testing them on three tasks:
1) To infer depth, object identity and location from images with occlusions. This is called the **inference task**.
2) To generate images with occlusions from pre-set representations of the objects in the images. This is called the  **graphics task**. 
3) Finally, I train a network that combines the two tasks into one, from images to a compressed representation and then back, 
to investigate whether a better representation emerges and whether the network learns an insightful way to represent depth 
and compute with occlusions. This is called the **unsupervised task**. 

In the unsupervised task, we wanted the network to explicitly understand occlusion as caused by depth. To do this,
we trained it to reconstruct its input image but with relative depths of the objects switched. The network can, therefore, 
be described as a depth-inverting autoencoder. 

Through these three tasks I show that recurrent connections are key for inferring object properties under occlusion, and for
generating images of objects under occlusion. In the unsupervised task I show that the autoencoder performs better in the inference
than the networks trained in a supervised way on the inference task, suggesting that the representation that emerged was better suited
than the pre-set one given the task. 

## Code

The code is split between "Datasets" which is used to create the images for the training and test set; and "Architectures",
which contains code for the the models, the training, the testing and anything else.

Datasets was based on and adapted from Spoerer et al., 2017's [code](https://github.com/cjspoerer/digitclutter). 
The basic structure of the files to run the models was instead adapted from [1Konny](https://github.com/1Konny/Beta-VAE) on github.

## Dependencies 

The datset code requires the following packages that should be downloaded using the standard Anaconda distribution.
* scipy
* numpy
* Pillow
* ImageMagick
* Pandas
* tqdm
* matplotlib
* imageio
* pytorch (needs to be compatible to GPU - see pytorch website)
* torch & torchvision (needs to be compatible to GPU - see pytorch website)
* cudatoolkit (needs to be compatible to GPU - see pytorch website)



## Usage

### Creating datasets

Shell file for creating dataset found in the Datsets folder and entitled:

```
wkst_dataset_create.sh
```

which contains:

```
python main_digit_create.py \
   --n_samples_train 100000 --n_samples_gnrl 10000 \
   --n_letters 2 --offset random_occluded --digit_colour_type b_w_e --linewidth 20 \
   --fontsize 180 --FILENAME /home/riccardo/Desktop/Data/100k_2digt_BWE_2 \
```
This code creates one folder (the "FILENAME"), within which two more folders are created, one for training images and one for testing images. The number of training and testing images are set by "n_samples_train" and "n_samples_gnrl" in that order. 

The number of digits in each image of both training and testing sets is set by 'n_letters'. 
"offset" can take five options: *fixed_occluded*, *fixed_unoccluded*, *random_unccluded*, *random_occluded*, and *hidden_traverse*. The last one is used to create the small set of images for the behaviour task (see Results in Thesis).

"digit_colour_type" can either take *b_w* or *b_w_e* to signify digits being either one black and one white (only works with n_digits=2), or each digit being black with a white edge (works with n_digits =>2). 

Finally, "linewidth" and "fontsize" are parameters describing the size of the edge around the digit and the size of the digit itself. When using the *b_w* paramter, I reccommend using linewidth = 5 and fontsize=220. When using *b_w_e* I reccommend using linewidth = 20 and fontsize = 180. These are not set numbers, however, and can be adapted to the researcher's preference.


### Training and testing ANNs

Within the Architecture folder are four other folders: 
* data_loaders: code for converting .png files (dataset) into torch arrays, and returning dataloaders to be used in ANN training (see Pytorch for more on Dataloaders). 
* models: contains various versions of code for models. The most recent and correct one is **BLT_models**.
* solvers: contains the main workhorse of the code: pulls together specified arguments, models, datasets etc to train and test the ANNs. Contains two scripts - one called **sup_solver** for supervised tasks and **unsup_solver** for unsupervised tasks. 
* run_files: contains all the .sh files for different tasks, datasets and GPU clusters/workstations.  


An example supervised .sh file is shown below:
```
python main_mod.py --train True --ckpt_name None --testing_method supervised_decoder \
    --encoder B --decoder B --sbd False --encoder_target_type depth_ordered_one_hot_xy \
    --n_filter 32 --n_rep 4 --kernel_size 4 --padding 1 \
    --optim_type Adam --batch_size 100 --lr 5e-3 --beta 0 \
    --max_epoch 80 --gather_step 500 --display_step 50 --save_step 10000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
    --output_dir /home/riccardo/Desktop/Experiments/decoder_sup/2_digts/bwe/arch/B  \
```

An example unsupervised .sh file is shown below:
```
python main_mod.py --train False --ckpt_name last --testing_method unsupervised --AE True \
     --encoder BLT --decoder BLT --freeze_decoder False --z_dim_bern 24 --z_dim_gauss 0  \
     --optim_type Adam  --lr 1e-3 --batch_size 100 \
     --max_epoch 100 --gather_step 500 --display_step 20 --save_step 5000 \
     --dset_dir /home/riccardo/Desktop/Data/100k_2digt_BWE_2/digts/ \
     --output_dir /home/riccardo/Desktop/Experiments/AE/Unfrozen/2_digts/BLT_BLT_depth_zdim24_2 \
```

The network can be set to train or test mode using the "--train" argument. It then looks within the "--output_dir/checkpoints" folder for checkpoint named the same as "--ckpt_name". If found, it will load that checkpoint and train/test using those parameters.

As the network trains on the data found in "--dset_dir", every "--display_step" iterations, the training and testing loss is shown along side other variables depending on the task. At each "--gather_step", task-specific data is collected to then be plotted once the training has finished. In "--save_step", the network parameters are saved, and, depending on the task, testing plots are generated and saved in the "--output_dir".  Training goes on for "--max_epoch" number of epochs. 

The "--testing_method" allows one to specify whether the task is supervised, in which case whether an "supervised_encoder" or "supervised_decoder" is used; or "unsupervised", in which case either a normal Autoencoder or a Variational Autoencoder can be used (set by "--AE" = T or F). The VAE can itself be adapted to encode gaussian units or bernoulli or both, set by the number in "--z_dim_gauss" and "--z_dim_bern". 

In the supervised cases, the representation used can be changed to five differet options, using the '--encoder_target_type' argument. The most common one is the "depth_ordered_one_hot_xy" that can be used for all tasks other than the decoder with the solid2 dataset, as that requires the "depth_black_white_xy_xy" encoding as input. The choice of repreestation will also affect the size of the code ("--z_dim"), however, this is taken care of within the solver code. 

The model itself can be specified using the "--encoder", "--decoder", "--n_filter 32" "--n_rep 4" "--kernel_size 4" and "--padding 1" arguments. The training itself can be controlled by the --optim_type",   "--lr", and "--batch_size". 

Details of what each argument means can also be found in the **main_mod.py** file within run_files. 



