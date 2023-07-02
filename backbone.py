import copy
import pickle
import sys
from itertools import chain
import numpy as np
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

opt = AugOptions().parse()  # get training options

# Dataset.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
print('The number of training images = %d' % dataset_size)

history_data = []
for i, data in enumerate(dataset):  # inner loop within one epoch

    model.set_input(data)  # unpack data from dataset and apply preprocessing
    model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

print('May be the force with you.')


