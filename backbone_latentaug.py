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

# # Parameters.
# params_space = {
#     'w_lpips': [10], # [1], [0, 0.1, 0.3, 1, 3, 10],
#     'w_pix': [0.1], # [3], [0, 0.1, 0.3, 1, 3, 10],
#     'w_latent': [0.001], # [0.1], [0, 0.001, 0.003, 0.01, 0.03, 0.1],
#     'w_disc' : [0.01], # [0.01],  [0, 0.01, 0.03, 0.1, 0.3, 1],
#     'p_thres': [0.0], # [0.0],
#     'opt_num_epochs': [12], # [3, 6, 9],
#     'opt_lr': [0.01], # [0.001, 0.003, 0.01, 0.03, 0.1],
# }
# # Parameters.
# params_space = {
#     'w_lpips': [10], # [1], [0, 0.1, 0.3, 1, 3, 10],
#     'w_pix': [0.1], # [3], [0, 0.1, 0.3, 1, 3, 10],
#     'w_latent': [0.001], # [0.1], [0, 0.001, 0.003, 0.01, 0.03, 0.1],
#     'w_disc' : [0.01], # [0.01],  [0, 0.01, 0.03, 0.1, 0.3, 1],
#     'p_thres': [0.0], # [0.0],
#     'opt_num_epochs': [9], # [3, 6, 9],
#     'opt_lr': [0.01], # [0.001, 0.003, 0.01, 0.03, 0.1],
# }

params_space = {
    'w_lpips': [10], # [1], [0, 0.1, 0.3, 1, 3, 10],
    'w_pix': [0.1], # [3], [0, 0.1, 0.3, 1, 3, 10],
    'w_latent': [0.001], # [0.1], [0, 0.001, 0.003, 0.01, 0.03, 0.1],
    'w_disc' : [0.01], # [0.01],  [0, 0.01, 0.03, 0.1, 0.3, 1],
    'p_thres': [0.0], # [0.0],
    'opt_num_epochs': [6], # [3, 6, 9],
    'opt_lr': [0.01], # [0.001, 0.003, 0.01, 0.03, 0.1],
}

n_imgs = 10000
# for index_exp in range(1):
for index_exp in range(1):

    print(f'Performing iteration: {index_exp}')
    params = copy.deepcopy(params_space)
    for key in params_space.keys():
        params_list = params_space[key]
        params[key] = random.choice(params_list)

    params['n_imgs'] = n_imgs
    print('Parameters.')
    print(params)

    opt = AugOptions().parse(args=params)   #opt = AugOptions().parse()  # get training options

    outdir = os.path.join(opt.checkpoints_dir, opt.name)
    outdirname_list = ['img', 'latent', 'img_aug', 'latent_aug']
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
        data_w = augment.get_latent_input()
        data_w_aug = augment.get_latent_output()

        if os.path.exists(os.path.join(outdir, 'img')):
            util_io.write_pickle(data, os.path.join(outdir, 'img', f'img_{i}'))

        if os.path.exists(os.path.join(outdir, 'latent')):
            util_io.write_pickle(data_w, os.path.join(outdir, 'latent', f'w_{i}'))

        if os.path.exists(os.path.join(outdir, 'img_aug')):
            util_io.write_pickle(data_aug, os.path.join(outdir, 'img_aug', f'img_aug_{i}'))

        if os.path.exists(os.path.join(outdir, 'latent_aug')):
            util_io.write_pickle(data_w_aug, os.path.join(outdir, 'latent_aug', f'w_aug_{i}'))

    stats_time = augment.stats_time[1:]
    print(np.mean(stats_time))

print('May be the force with you.')


