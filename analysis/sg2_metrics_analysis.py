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

import pickle
import json

import dnnlib
from metrics import metric_main_mi_multimodal
from utils import util_general
from utils import util_path

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

num_gpus =1
rank = 0
device = 'cuda:0'

metrics_name = [ 'fid50k_full', 'pr50k3_full']
metrics_cache = True # Options to upload the cache real features for FID

# Create path to StyleGAN.
model_dir = './models/'
report_dir = './reports/'
exp_name = 'aug_r' #'aug_pr_ablation' # 'aug_pr'
dataset = 'Pelvis_2.1_repo_no_mask'
run_dir = os.path.join(report_dir, dataset, exp_name)
util_path.create_dir(run_dir)
dataset_name = 'Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10'
modalities = "MR_nonrigid_CT,MR_MR_T2"
modalities = (modalities.replace(" ", "").split(",")) # Convert string to list.
stylegan_dir = os.path.join(
        model_dir, dataset, "training-runs", dataset_name,  util_general.parse_separated_list_comma(modalities)
)
stylegan_exp_name = '00003'
stylegan_pkl_name = 'network-snapshot-005320.pkl'

# Load StyleGANAN
snapshot_data = load_stylegan(stylegan_dir, stylegan_exp_name, stylegan_pkl_name)
G_kwargs = {'truncation_psi': 1.0}

# Load training opt (contains the path to the training data)
dataset_kwargs = dnnlib.EasyDict(snapshot_data['training_set_kwargs'])
dataset_kwargs['path'] = './data/interim/Pelvis_2.1_repo_no_mask/Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10.zip'

n_img_aug = 10000
exps = [x for x in os.listdir(os.path.join(report_dir, dataset)) if f'augment_0-n_imgs_{n_img_aug}' in x]
#exps = [x for x in os.listdir(os.path.join(report_dir, dataset)) if f'augment-n_imgs_{n_img_aug}' in x]
#exps = [x for x in os.listdir(os.path.join(report_dir, dataset)) if f'augment_ablation-n_imgs_{n_img_aug}-' in x]

assert len(exps) != 0

result_dict_modes = util_general.list_dict()
for exp in exps:
    print('\n')
    print(exp)
    try:
        aug_name, aug_n_imgs, aug_opt = exp.split(sep='-')
    except ValueError:
        t = exp.split('-')
        aug_name = t[0]
        aug_n_imgs = t[1]
        aug_opt = exp.split(aug_n_imgs+'-')[-1]
    aug_basename = aug_name + '-' + aug_n_imgs + '-' + aug_opt
    #dataset_kwargs_gen = dnnlib.EasyDict({
    #    'data_dir': os.path.join(report_dir, dataset),
    #    'aug_name': aug_name,
    #    'aug_opt': aug_opt,
    #    'n_imgs': aug_n_imgs,
    #    'batch_size': 16
    #})
    dataset_kwargs_gen = dnnlib.EasyDict({
        'aug_name': aug_name,
        'dataroot': os.path.join(report_dir, dataset,exp),
        'n_imgs': int(aug_n_imgs.split('_')[-1]),
        'batch_size': 16
    })

    # Evaluate metrics.
    for metric in metrics_name:
        print(metric)
        for idx_mode, mode in enumerate(modalities):
            result_dict_mode = metric_main_mi_multimodal.calc_metric(
                metric=metric, G=snapshot_data['G_ema'], dataset_kwargs=dataset_kwargs, dataset_kwargs_gen=dataset_kwargs_gen, G_kwargs=G_kwargs, num_gpus=num_gpus, rank=rank, device=device,  cache=metrics_cache, mode_dict={'mode_name': mode, 'mode_idx': idx_mode},
            )
            print(f"{mode} : {result_dict_mode.results}")
            temp = {
                'metric': result_dict_mode['metric'],
                'mode': mode,
                'value': result_dict_mode.results
            }
            result_dict_modes[aug_basename].append(temp)

    jsonl_line = json.dumps(result_dict_modes, indent=3)
    with open(os.path.join(run_dir, f'metric-{exp}.jsonl'), 'at') as f:
        f.write(jsonl_line + '\n')

jsonl_line = json.dumps(result_dict_modes, indent=3)
with open(os.path.join(run_dir, f'metric.jsonl'), 'at') as f:
    f.write(jsonl_line + '\n')

print('May be the force with you.')
