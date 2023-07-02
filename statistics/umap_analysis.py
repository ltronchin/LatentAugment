import os
import sys
sys.path.extend([
    "./",
    "./src/models/stylegan3/"
])
my_env = os.environ.copy()
my_env["PATH"] = "/home/lorenzo/miniconda3/envs/latentaugment/bin:" + my_env["PATH"]
os.environ.update(my_env)

import pickle
from tqdm import  tqdm
import numpy as np
import umap
import torch

from augments.utils import util_dataset
from utils import util_path
from utils import util_reports

def fromdir_tolist(datadir):
    data = []
    files = os.listdir(datadir)
    files = [x for x in files if not x.startswith('.')]
    files.sort()
    for i, file in enumerate(files):
        filename = os.path.join(datadir, file)
        with open(filename, 'rb') as f:
            x = pickle.load(f)
            data.append(x)

    return data

# Parameters
dataset = 'Pelvis_2.1_repo_no_mask'
dataset_w_name = 'Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10-expinv_00001'
checkpoints_dir = '/home/lorenzo/LatentAugment/reports/'
interim_dir = '/home/lorenzo/LatentAugment/data/interim/'
# the two experiments to compare
exp_latent_augment =  'latent_augment_umap-n_imgs_10000-opt_lr_0.01-opt_num_epochs_12-w_latent_0.001-w_pix_0.1-w_lpips_10-w_disc_0.01' # 'latent_augment_umap-n_imgs_10000-opt_lr_0.01-opt_num_epochs_6-w_latent_0.01-w_pix_0.3-w_lpips_3-w_disc_0.3'
exp_randomgan_augment = 'random_augment_umap-n_imgs_10000-truncation_psi_1.0'
dump_to_disk = True

mode = 'CT' #CT
if mode=='MRI':
    analysis_augment = 'analysis_augment_MRI_umap' # 'analysis_augment_MRI_umap'
    mode_name = 'B'
elif mode=='CT':
    analysis_augment = 'analysis_augment_CT_umap'# 'analysis_augment_CT_umap'
    mode_name = 'A'
else:
    raise NotImplementedError

batch_size = 16

w_dim = 512
num_ws = 14
n_imgs = 100
full_inverted_set = True # fit umap on the latent representation of the inverted set.

util_path.create_dir(os.path.join(checkpoints_dir, dataset, analysis_augment))
# Load data from the disk and put into lists.
# Real data
data = fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_latent_augment, 'img')) # Real images.
latents = fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_latent_augment, 'latent')) # Latent code Real images.
# Latent augment
data_aug_latent = fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_latent_augment, 'img_aug')) # Augmented images.
latents_aug_latent = fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_latent_augment, 'latent_aug')) # Latent code augmented images.
# Random augment
data_aug_randomgan =  fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_randomgan_augment, 'img_aug')) # Augmented Images.
latents_aug_randomgan =  fromdir_tolist(os.path.join(checkpoints_dir, dataset, exp_randomgan_augment, 'latent_aug')) # Latent code augmented images.

# Convert lists to numpy.
# Real data
imgs = []
for img in data:
    x_A = img[mode_name].squeeze(dim=1)
    x_A = x_A.cpu().detach().numpy()
    imgs.append(x_A)
imgs = np.array(imgs).reshape((len(data) * batch_size, 256, 256))
# Latent
w = [x['w'] for x in latents]
w = np.array(w).reshape((len(latents) * batch_size, w_dim))
y = np.zeros(w.shape[0], dtype=int)

# Latent Augment
imgs_latent = []
for img in data_aug_latent:
    x_A = img[mode_name].squeeze(dim=1)
    x_A = x_A.cpu().detach().numpy()
    imgs_latent.append(x_A)
imgs_latent = np.array(imgs_latent).reshape((len(data_aug_latent) * batch_size, 256, 256))
# Latent
w_latent = [x['w'] for x in latents_aug_latent]
w_latent = np.array(w_latent).reshape((len(latents_aug_latent) * batch_size, w_dim))
y_latent = np.ones(w_latent.shape[0], dtype=int)

# Random Augment
imgs_randomgan = []
for img in data_aug_randomgan:
    x_A = img[mode_name].squeeze(dim=1)
    x_A = x_A.cpu().detach().numpy()
    imgs_randomgan.append(x_A)
imgs_randomgan = np.array(imgs_randomgan).reshape((len(data_aug_randomgan) * batch_size, 256, 256))
# Latent
w_randomgan = [x['w'] for x in latents_aug_randomgan]
w_randomgan = np.array(w_randomgan).reshape((len(latents_aug_randomgan) * batch_size, w_dim))
y_randomgan = np.full(w_randomgan.shape[0], 2, dtype=int)

# Plot images.
if dump_to_disk:
    util_reports.dump_images(os.path.join(checkpoints_dir, dataset, analysis_augment), imgs[:n_imgs], 'img')
    util_reports.dump_images(os.path.join(checkpoints_dir, dataset, analysis_augment), imgs_latent[:n_imgs], 'img_latent')
    util_reports.dump_images(os.path.join(checkpoints_dir, dataset, analysis_augment), imgs_randomgan[:n_imgs], 'img_randomgan')

# Upload the inverted training set.
if full_inverted_set:
    stats_dataset_w = util_dataset.LatentCodeDataset(
                path=os.path.join(interim_dir, dataset, dataset_w_name + '.zip'), split='train', w_dim=w_dim, num_ws=num_ws
    )
    data_loader = torch.utils.data.DataLoader(dataset=stats_dataset_w, batch_size=1, shuffle=False)
    with tqdm(total=len(data_loader.dataset), unit=' img/latent/fea ') as pbar:
        stats = util_dataset.DatasetStats(manifold='latent', max_items=100000, step=5)
        for x, fname in data_loader:
            added_shape = stats.append_torch(x, fname)

            if added_shape < 0:
                break
            pbar.update(x.shape[0])
    stats_all = stats.get_all()
    W = [x[0] for x in stats_all]
    W = np.array(W).reshape((len(W), w_dim))
    # Fit umap.
    reducer = umap.UMAP(random_state=42)
    transformer = reducer.fit(W)
else:
    reducer = umap.UMAP(random_state=42)
    transformer = reducer.fit(w)

# Project latent points.
w_to_project = np.concatenate((w[:n_imgs], w_latent[:n_imgs], w_randomgan[:n_imgs]))
y_to_project = np.concatenate((y[:n_imgs], y_latent[:n_imgs], y_randomgan[:n_imgs]))
imgs_to_project =  np.concatenate((imgs[:n_imgs], imgs_latent[:n_imgs], imgs_randomgan[:n_imgs]))
embedding = transformer.transform(w_to_project)

if dump_to_disk:
    with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'embedding_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'wb') as handle:
        pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'y_to_project_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'wb') as handle:
        pickle.dump(y_to_project, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'imgs_to_project_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'wb') as handle:
        pickle.dump(imgs_to_project, handle, protocol=pickle.HIGHEST_PROTOCOL)

util_reports.scatter_plot(os.path.join(checkpoints_dir, dataset, analysis_augment), embedding, y_to_project, f'umap_reduced_full_inverted_set_{full_inverted_set}')
util_reports.scatter_plot_interactive(os.path.join(checkpoints_dir, dataset, analysis_augment), embedding, y_to_project, imgs_to_project, f'umap_reduced_bokeh_full_inverted_set_{full_inverted_set}')

print('May be the force with you.')