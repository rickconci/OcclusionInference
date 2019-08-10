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





