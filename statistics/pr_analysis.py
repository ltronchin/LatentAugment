import sys
import copy
sys.path.extend([
    "./",
    "./src/models/stylegan3/"
])
import os
my_env = os.environ.copy()
my_env["PATH"] = "/home/lorenzo/miniconda3/envs/latentaugment/bin:" + my_env["PATH"]
os.environ.update(my_env)

import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import numpy as np
import cv2
import torch

from  utils import util_path


# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
#sns.set_context("paper")
#sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Code for seaborn blob plot
#
# import seaborn as sns
# sns.set_context("paper")
# sns.set_theme(style="ticks")
# scatter = sns.relplot(data=df, x="Recall", y="Precision", size="FID", style="Aug_method", hue="Aug_method",
#                       palette=["limegreen", "lightgray"], markers=["o", "^"], alpha=0.8, sizes=(1, 300))
# for line in range(0, df.shape[0]):
#     plt.text(df.Recall[line] + 0.01, df.Precision[line], df.Exp_ID[line], horizontalalignment='left',
#              size='xx-small', color='black', weight='light')
# scatter.ax.grid(True, linewidth=0.1)
# scatter.savefig(os.path.join(run_dir, f"prfid_{mode}.png"), dpi=400, format='png')
# plt.show()

augment_symbols = {'LatentAugment': 'o', 'RandomAugment': '^'}
augment_colours = {'LatentAugment': [0 / 255, 150 / 255, 136 / 255], 'RandomAugment':'white'} #augment_colours = {'LatentAugment': 'limegreen', 'RandomAugment': 'lightgray'}
augment_alphas = {'LatentAugment': 1.0, 'RandomAugment': 1.0}
augment_label = {'LatentAugment': 'LatentAugment', 'RandomAugment': 'Standard SG2 DA'}
modalities_label = {'MR_nonrigid_CT': 'CT', 'MR_MR_T2': 'MRI'}
dump_pr=True
dump_grid=False
bubble_name= '' #'fid'

def generate_mapping(n):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mapp = {}
    for i in range(n):
        mapp[i] = alphabet[i]
    return mapp

report_dir = './reports/'
exp_name = 'aug_pr_0' # 'aug_r' # 'aug_pr_ablation'# 'aug_pr'
dataset = 'Pelvis_2.1_repo_no_mask'
run_dir = os.path.join(report_dir, dataset, exp_name)
modalities = "MR_nonrigid_CT,MR_MR_T2"
modalities = (modalities.replace(" ", "").split(",")) # Convert string to list.

with open(os.path.join(run_dir, 'metric.jsonl'), 'rb') as f:
    data = json.load(f)
exps = sorted(list(data.keys()))
mapping = generate_mapping(len(exps))
exps_mapping = {exps[i]: mapping[i] for i in range(len(exps))}

columns = ['Exp_ID', 'Exp_name', 'Aug_method', 'Precision', 'Recall', 'FID']
for mode in modalities:
    df = pd.DataFrame(columns=columns)
    for exp in exps:
        exp_data = data[exp]
        exp_data_mode_pr = [x for x in exp_data if x['mode'] == mode and x['metric'] == 'pr50k3_full']
        exp_data_mode_fid = [x for x in exp_data if x['mode'] == mode and x['metric'] == 'fid50k_full']
        assert len(exp_data_mode_pr) == 1
        row = {
            'Exp_ID': exps_mapping[exp],
            'Exp_name': exp,
            #'Aug_method': 'LatentAugment' if exp.split('-')[0] == 'latent_augment' else 'RandomAugment',
            'Aug_method': 'LatentAugment' if exp.split('-')[0] == 'latent_augment_0' else 'RandomAugment',
            'Precision': exp_data_mode_pr[0]['value']['pr50k3_full_precision'],
            'Recall': exp_data_mode_pr[0]['value']['pr50k3_full_recall'],
            'FID': exp_data_mode_fid[0]['value']['fid50k_full'],
            'MAE': [],
        }
        df = df.append(row, ignore_index=True)

    df.to_excel(os.path.join(run_dir, f"pr_{mode}.xlsx"))

    if dump_pr:
        fig, ax = plt.subplots()
        for method, group in df.groupby('Aug_method'):
            symbol = augment_symbols[method]
            color = augment_colours[method]
            alpha = augment_alphas[method]

            if bubble_name == 'fid':
                ax.scatter(group['Recall'], group['Precision'], s=list(group['FID'].values*2), c=color, edgecolor='black', marker=symbol,  alpha=alpha, label=augment_label[method])
            elif bubble_name == 'mae':
                ax.scatter(group['Recall'], group['Precision'], s=group['FID']*2, c=color, edgecolor='black',  marker=symbol, alpha=alpha, label=augment_label[method])
            else:
                ax.scatter(group['Recall'], group['Precision'], c=color, edgecolor='black', marker=symbol, s=50, alpha=alpha, label=augment_label[method])

        for x, y, label in zip(df['Recall'], df['Precision'], df['Exp_ID']):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords='offset points', fontsize=12)

        # set labels and legend
        ax.set_xlabel('Recall (diversity)')
        ax.set_ylabel('Precision (fidelity)')
        #ax.set_xlim([0,0.8])
        #ax.set_ylim([0,0.8])
        ax.legend()
        ax.grid(linestyle='-', linewidth=0.1)
        plt.tight_layout()
        if bubble_name == 'fid':
            #plt.title(f'Precision-recall-FID {modalities_label[mode]}')
            fig.savefig(os.path.join(run_dir, f"prfid_{mode}.pdf"), dpi=400, format='pdf',bbox_inches='tight')
        elif bubble_name == 'mae':
            #plt.title(f'Precision-recall-MAE {modalities_label[mode]}')
            fig.savefig(os.path.join(run_dir, f"prmae_{mode}.pdf"), dpi=400, format='pdf',bbox_inches='tight')
        else:
            plt.title(f'{modalities_label[mode]}')
            fig.savefig(os.path.join(run_dir, f"pr_{mode}.pdf"), dpi=400, format='pdf', bbox_inches='tight')

        plt.show()

    if dump_grid:
        n_to_plot=6
        for exp in exps:
            for img_flag in ['img', 'img_aug']:
                img_filenames = sorted(util_path.listdir_nohidden_with_path(os.path.join(report_dir, dataset, exp, img_flag)))
                imgs_npy = []
                for img_filename in img_filenames[:n_to_plot]:
                    print(img_filename)
                    with open(img_filename, 'rb') as f:
                         images = pickle.load(f)

                    if mode == 'MR_nonrigid_CT':
                        x = images['A']
                    elif mode == 'MR_MR_T2':
                        x = images['B']
                    else:
                        raise NotImplementedError

                    x = ((x[0] / 1.2) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    x_npy = x[0].cpu().detach().numpy()
                    imgs_npy.append(x_npy)

                rows = [cv2.hconcat(imgs_npy[i : i + 2]) for i in range(0, len(imgs_npy), 2)]
                grid = cv2.vconcat(rows)
                cv2.imwrite(os.path.join(run_dir, f'{exp}_{mode}_{img_flag}_grid.png'), grid)

print('May be the force with you.')