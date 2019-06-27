"""main.py"""

import argparse

import numpy as np
import torch

from solver_mod import Solver
from utils_mod import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    net = Solver(args)
    if args.train:
        print("Training")
        net.train()
        print("Testing")
        net.test()
        
    elif args.train==False:
        print("Testing")
        net.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OcclusionInference')

    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--seed', default=2019, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=400, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--z_dim', default=20, type=int, help='dimension of the representation z')
    parser.add_argument('--n_filter', default=32, type=int, help='number of filters in convolutional layers')

    parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--model', default='conv_VAE_32', type=str, help='conv_VAE_32 or conv_VAE_64')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='/train/', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='digits', type=str, help='dataset name')

    parser.add_argument('--image_size', default=32, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=False, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=5, type=int, help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=10, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=200, type=int, help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='None', type=str, help='load previous checkpoint. insert checkpoint filename')

    args = parser.parse_args()

    main(args)
