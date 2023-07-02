import copy
import pickle
import sys
from itertools import chain
import numpy as np
import copy
import random
sys.path.extend([
    "./",
    "./src/models/stylegan3/"
])
import os
my_env = os.environ.copy()
my_env["PATH"] = "/home/lorenzo/miniconda3/envs/latentaugment/bin:" + my_env["PATH"]
os.environ.update(my_env)

from options.aug_options import AugOptions

from augments import create_augment
from data import create_dataset

from utils import util_path
from utils import util_io

truncation_psi =  [1.0] # [0.0, 0.3, 0.7, 1.0] # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
n_imgs = 10000
for psi_value in truncation_psi:

    params = {
        'n_imgs': n_imgs,
        'p_thres': 0.0,
        'truncation_psi': psi_value
    }
    print('Parameters.')
    print(params)

    opt = AugOptions().parse(args=params)   #opt = AugOptions().parse()  # get training options

    outdir = os.path.join(opt.checkpoints_dir, opt.name)
    outdirname_list = ['img', 'latent', 'img_aug', 'latent_aug'] # ['img', 'img_aug', 'latent_aug'] #  ['img', 'latent', 'img_aug', 'latent_aug']
    for outname in outdirname_list:
        util_path.create_dir(os.path.join(outdir, outname))

    # Dataset.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Augment.
    augment = create_augment(opt)  # define the augmentation pipeline for LatentAugment.

    # Sanity check for augmentation class.
    iterator = iter(dataset)
    data = next(iterator)
    augment.set_input(data)
    augment.sanity_check()

    n_iter = n_imgs // opt.batch_size
    for i, data in enumerate(dataset):  # inner loop within one epoch

        print(f"Iteration: {i} of {n_iter}")
        if i >= n_iter :
            break

        # Set input for augmentation.
        augment.set_input(data)

        # Perform the augmentation.
        augment.forward()

        # Get output from augmentation.
        # Img
        data_aug = augment.get_output()
        # Latents
        data_w_aug = augment.get_latent_output()

        if os.path.exists(os.path.join(outdir, 'img')):
            util_io.write_pickle(data, os.path.join(outdir, 'img', f'img_{i}'))

        if os.path.exists(os.path.join(outdir, 'img_aug')):
            util_io.write_pickle(data_aug, os.path.join(outdir, 'img_aug', f'img_aug_{i}'))

        if os.path.exists(os.path.join(outdir, 'latent_aug')):
            util_io.write_pickle(data_w_aug, os.path.join(outdir, 'latent_aug', f'w_aug_{i}'))

    stats_time = augment.stats_time[1:]
    print(np.mean(stats_time))

print('May be the force with you.')


