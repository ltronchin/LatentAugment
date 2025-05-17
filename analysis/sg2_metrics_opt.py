'''
Optimization of the parameters of the LatentAugment algorithm using Optuna.
'''

import sys
import copy
sys.path.extend([
    "./",
    "./src/",
    "./src/models/stylegan3/"
])
import os
my_env = os.environ.copy()
my_env["PATH"] = "/home/lorenzo/miniconda3/envs/latentaugment/bin:" + my_env["PATH"]
os.environ.update(my_env)
import pickle

import json
import shutil
import optuna
import numpy as np
import pandas as pd

# For metrics
import dnnlib
from metrics import metric_main_mi_multimodal
# For imgs generation.
from options.aug_options import AugOptions
from augments import create_augment
from data import create_dataset

from utils import util_general
from utils import util_path
from utils import util_io

def load_stylegan(dir_model, exp_stylegan, network_pkl_stylegan, load_opt='dict'):

    exp_name = [x for x in os.listdir(dir_model) if exp_stylegan in x]  # search for model
    assert len(exp_name) == 1
    path = os.path.join(dir_model, exp_name[0], network_pkl_stylegan)

    # Load the network.
    print(f'Loading stylegan from "{path}"...')
    with open(path, 'rb') as f:
        network_dict = pickle.load(f)
    print('Done.')

    if load_opt == 'networks':
        G = network_dict['G_ema']  # subclass of torch.nn.Module
        D = network_dict['D']

        G = G.eval().requires_grad_(False)
        D = D.eval().requires_grad_(False)

        return G, D
    elif load_opt == 'dict':
        return network_dict
    else:
        raise NotImplementedError


def dump_imgs(trial, n_imgs=10000):

    # Define parameters space.
    w_lpips = trial.suggest_categorical("w_lpips", [0, 0.1, 0.3, 1, 3, 10])
    w_pix = trial.suggest_categorical("w_pix", [0, 0.1, 0.3, 1, 3, 10])
    w_latent = trial.suggest_categorical("w_latent", [0, 0.001, 0.003, 0.01, 0.03, 0.1])
    w_disc = trial.suggest_categorical("w_disc", [0, 0.01, 0.03, 0.1, 0.3, 1])
    p_thres = 0.0 # fixed
    opt_num_epochs = trial.suggest_categorical("opt_num_epochs", [3, 6, 9])
    opt_lr = trial.suggest_categorical("opt_lr", [0.001, 0.003, 0.01, 0.03, 0.1])

    params = {
        'w_lpips': w_lpips,
        'w_pix': w_pix,
        'w_latent': w_latent,
        'w_disc': w_disc,
        'p_thres': p_thres,
        'opt_num_epochs': opt_num_epochs,
        'opt_lr': opt_lr,
        'init_w': 'inv',
        'n_imgs': n_imgs
    }
    opt = AugOptions().parse(args=params)  # opt = AugOptions().parse()  # get training options

    outdir = os.path.join(opt.checkpoints_dir, opt.name)
    util_path.create_dir(os.path.join(outdir, 'img_aug'))

    # Training dataset.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Augment.
    augment = create_augment(opt)  # define the augmentation pipeline for LatentAugment.

    n_iter = n_imgs // opt.batch_size
    for i, data in enumerate(dataset):  # inner loop within one epoch

        print(f"Iteration: {i} of {n_iter}")
        if i >= n_iter :
            break

        # Set input for augmentation.
        augment.set_input(data)
        augment.forward()
        data_aug = augment.get_output()
        util_io.write_pickle(data_aug, os.path.join(outdir, 'img_aug', f'img_aug_{i}'))

    return opt

def calc_pr(opt, metrics_name=None):

    if metrics_name is None:
        metrics_name = ['fid50k_full', 'pr50k3_full']
    exp = opt.name
    run_dir = opt.checkpoints_dir
    synthetic_dir = os.path.join(opt.checkpoints_dir, opt.name)
    num_gpus = 1
    device = f"cuda:{opt.gpu_ids[0]}"
    rank = 0
    metrics_cache = True # Options to upload the cache real features for FID

    # Create path to StyleGAN.
    model_dir = opt.model_dir
    dataset = opt.dataset_aug
    dataset_name = opt.dataset_name_aug
    modalities = opt.modalities
    modalities = (modalities.replace(" ", "").split(","))  # Convert string to list.

    stylegan_dir = os.path.join(
        model_dir, dataset, "training-runs", dataset_name, util_general.parse_separated_list_comma(modalities)
    )
    # Load StyleGAN
    stylegan_exp_name = opt.exp_stylegan
    stylegan_pkl_name = opt.network_pkl_stylegan
    snapshot_data = load_stylegan(stylegan_dir, stylegan_exp_name, stylegan_pkl_name)
    G_kwargs = {'truncation_psi': opt.truncation_psi}

    # Load training opt (contains the path to the training data)
    dataset_kwargs_sg2 = False
    if dataset_kwargs_sg2:
        dataset_kwargs = dnnlib.EasyDict(snapshot_data['training_set_kwargs'])
        dataset_kwargs['path'] = opt.dataroot
    else:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_mi_multimodal.CustomImageFolderDataset',
                                         path=opt.dataroot,
                                         dtype='float32',
                                         split='val',
                                         max_size = None,
                                         modalities=modalities,
                                         use_labels=False,
                                         xflip=True
    )
    # Synthetic data opt
    dataset_kwargs_gen = dnnlib.EasyDict({
        'aug_name': f'{opt.aug}_augment',
        'dataroot': synthetic_dir,
        'n_imgs': opt.n_imgs,
        'batch_size': opt.batch_size
    })

    result_dict_modes = util_general.list_dict()
    for metric in metrics_name:
        print(metric)
        for idx_mode, mode in enumerate(modalities):
            result_dict_mode = metric_main_mi_multimodal.calc_metric(
                metric=metric,
                G=snapshot_data['G_ema'],
                dataset_kwargs=dataset_kwargs,
                dataset_kwargs_gen=dataset_kwargs_gen,
                G_kwargs=G_kwargs,
                num_gpus=num_gpus,
                rank=rank,
                device=device,
                cache=metrics_cache,
                mode_dict={'mode_name': mode, 'mode_idx': idx_mode},
            )
            print(f"{mode} : {result_dict_mode.results}")
            temp = {
                'metric': result_dict_mode['metric'],
                'mode': mode,
                'value': result_dict_mode.results
            }
            result_dict_modes[exp].append(temp)

    jsonl_line = json.dumps(result_dict_modes, indent=3)
    with open(os.path.join(run_dir, f'metric-{exp}.jsonl'), 'at') as f:
        f.write(jsonl_line + '\n')

    precision = np.mean([x['value']['pr50k3_full_precision'] for x in result_dict_modes[exp] if x['metric'] == 'pr50k3_full'])
    recall = np.mean([x['value']['pr50k3_full_recall'] for x in result_dict_modes[exp] if x['metric'] == 'pr50k3_full'])

    # Remove the directory containing the synthetic data to save space.
    shutil.rmtree(synthetic_dir)

    return precision, recall

def objective(trial):

    opt = dump_imgs(trial, n_imgs=10000) # n_imgs=29842 equal to the len of training set
    precision, recall = calc_pr(opt)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

def objective_recall(trial):

    opt = dump_imgs(trial, n_imgs=10000) # n_imgs=29842 equal to the len of training set
    _, recall = calc_pr(opt)

    return recall

def objective_precision(trial):
    opt = dump_imgs(trial, n_imgs=10000)  # n_imgs=29842 equal to the len of training set
    precision, _ = calc_pr(opt)

    return precision

if __name__ == '__main__':
    # checkpoint_dir = "./reports/Pelvis_2.1_repo_no_mask/aug_pr_optuna_val"
    # checkpoint_dir = "./reports/Pelvis_2.1_repo_no_mask/aug_r_optuna_val"
    checkpoint_dir = "./reports/CESM_dataset/aug_r_optuna_val"
    try:
        with open(os.path.join(checkpoint_dir, 'optuna_study.pickle'), 'rb') as handle:
            study = pickle.load(handle)
        print('load study.')
    except FileNotFoundError:
        study = optuna.create_study(directions=["maximize"])
        print('create new study.')

    # study.optimize(objective, n_trials=65)
    study.optimize(objective_recall, n_trials=50)

    best_params = study.best_params
    print(best_params)

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df.to_excel(os.path.join(checkpoint_dir, 'optuna_study.xlsx'))
    with open(os.path.join(checkpoint_dir, 'optuna_study.pickle'), 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('May be the force with you.')
