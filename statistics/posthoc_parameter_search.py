import sys
import copy
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([
    "./",
    "./src/models/stylegan3/"
])

import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
import re
from re import search

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 16

from genlib.utils import util_general

# Directory.
# opt_source_dir = "./reports/Pelvis_2.1_repo_no_mask/training-runs-pix2pix/feature_importance_pix2pix/"
opt_source_dir = "./reports/Pelvis_2.1_repo_no_mask/training-runs-pix2pix/feature_importance_cycle_gan/"
# Parameters
# xname = ['p_thres', 'opt_lr', 'opt_num_epochs', 'w_latent', 'w_pix', 'w_lpips', 'w_disc', 'lower_bound_clip']
xname = ['p_thres', 'opt_lr', 'opt_num_epochs', 'w_latent', 'w_pix', 'w_lpips', 'w_disc']
yname = ['mae_500hu']
iterations = 100
regressor_list = [('random_forest', RandomForestRegressor(n_estimators=100))]

# Load report file
history = pd.read_excel(os.path.join(opt_source_dir, 'overall_history_mean.xlsx'))

# Create training dataset
# X - > matrix of parameters space 'p_thres', 'opt_lr', 'opt_num_epochs', 'w_latent', 'w_pix', 'w_lpips', 'w_disc'
# y - > mae_500hu
# Drop columns out of research space.
[history.drop(columns=[key], inplace=True) for key in history.keys() if not (key in xname or key in yname)]
try:
    history['lower_bound_clip'] = history['lower_bound_clip'].map({True: 1, False: 0})
except KeyError:
    pass

# Create dataset.
X = history.iloc[:, :-1]
y = history.iloc[:, -1]

scaler = StandardScaler()
scaler.fit(X)
print(scaler.mean_)
X = scaler.transform(X)

fimp_regressor = {}
for rfr_name, rfr in regressor_list:
    print(rfr_name)
    fimp = []
    for i in range(iterations):
        print(f'iter {i}')
        rfr.fit(X, y)
        coef = rfr.feature_importances_
        fimp.append(list(coef))

    fimp = np.asarray(fimp)
    fimp_df = pd.DataFrame(fimp, columns=xname)
    fimp_df.to_excel(os.path.join(opt_source_dir, 'fimp_random_forest.xlsx'))
    fmean = fimp.mean(axis=0)
    fstd = fimp.std(axis=0)

    importances = pd.Series(fmean, index=xname)
    fig, ax = plt.subplots()
    importances.plot.bar(yerr=fstd, ax=ax, color='red')
    ax.set_title(f"{rfr_name} feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(os.path.join(opt_source_dir, f"{rfr_name}_feature_importance.png"), dpi=400, format='png')
    plt.show()

fig = px.parallel_coordinates(
    history, color="mae_500hu",  color_continuous_scale=px.colors.diverging.Tealrose,  color_continuous_midpoint=7)
fig.show()

print('May be the force with you.')