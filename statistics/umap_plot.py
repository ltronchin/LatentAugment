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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import seaborn as sns
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters
dataset = 'Pelvis_2.1_repo_no_mask'
checkpoints_dir = './reports/'
n_imgs = 100
full_inverted_set = True

mode = 'CT' #CT
if mode=='MRI':
    analysis_augment = 'analysis_augment_MRI_umap'
elif mode=='CT':
    analysis_augment = 'analysis_augment_CT_umap'
else:
    raise NotImplementedError

with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'embedding_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'rb') as handle:
    embedding = pickle.load(handle)
with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'y_to_project_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'rb') as handle:
    y_to_project = pickle.load(handle)
with open(os.path.join(checkpoints_dir, dataset, analysis_augment, f'imgs_to_project_{n_imgs}-full_inverted_set_{full_inverted_set}.pickle'), 'rb') as handle:
     imgs_to_project = pickle.load(handle)

output_name = f'umap_reduced_{n_imgs}-full_inverted_set_{full_inverted_set}'
output_dir = os.path.join(checkpoints_dir, dataset, analysis_augment)
# Plot the scatter plot.
labels_name = ['Real data', 'LatentAugment samples', 'Standard SG2 DA samples']
alphas = [1.0, 1.0, 1.0]
sizes = [50, 50, 50]
markers = ['*', 'o', '^']
colors = [
    [30 / 255, 136 / 255, 229 / 255],
    [0 / 255, 150 / 255, 136 / 255],
    [255 / 255, 255 / 255, 255 / 255]]  # blue, green, white

fig, ax = plt.subplots()
# Plot each label separately with the appropriate color, marker, size, and alpha.
for i in range(len(np.unique(y_to_project))):
    mask = (y_to_project == i)
    ax.scatter(
        # embedding[mask, 0], embedding[mask, 1], c=colors[i], edgecolor='black', marker=markers[i], s=sizes[i], alpha=alphas[i],
        embedding[mask, 0], embedding[mask, 1], color=colors[i], edgecolor='black', marker=markers[i], s=sizes[i], alpha=alphas[i],
        label=labels_name[i]
    )
# Add axis labels and title.
plt.xlabel('Embedding 1')
plt.ylabel('Embedding 2')
plt.legend()

fig.savefig(os.path.join(output_dir, f"{output_name}.pdf"), dpi=400, format='pdf',  bbox_inches='tight')
# Show the plot.
plt.show()

print('May be the force with you.')