import copy
import os
# Added library with respect to template
import pickle
import random
import time
import json

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from augments.utils import util_latent_aug
from utils import util_path
from .base_aug import BaseAugment

#  -----------------------------------------------------------------------------------------------------------------
# Helper.
def reverse_broadcasting(latent):
    return latent[:, :1, :]  # Reverse the broadcasting operation [batch_size, 14, 512] -> [batch_size, 1, 512]

def map_range(x, old_min=-1000, old_max=2000, new_min=-1, new_max=1):

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    x = (((x - old_min) * new_range) / old_range) + new_min

    return x

def set_gpu_ids(gpu_ids):

    # set gpu ids
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            gpu_ids.append(gpu_id)

    return gpu_ids

class LatentAugment(BaseAugment):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            parser:   -- original option parser
        Returns:
            the modified parser.

        """

        parser.add_argument('--model_dir', help='Where to load the StyleGAN/MappingNetwork pretrained model', metavar='DIR', required=True)
        parser.add_argument('--interim_dir', help='Where to save/load the data', metavar='DIR', required=True)
        parser.add_argument('--gpu_ids_aug', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # Common dataset options.
        parser.add_argument('--dataset_aug', help='', metavar='DIR', default="Pelvis_2.1_repo_no_mask")
        parser.add_argument('--dataset_name_aug', help='', metavar='DIR',  default="Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10")
        parser.add_argument('--modalities_aug', help='', metavar='DIR', default="MR_nonrigid_CT,MR_MR_T2")
        parser.add_argument('--img_resolution',  help='Image resolution.', type=int, default=256)
        # StyleGAN options.
        parser.add_argument('--exp_stylegan', help='', metavar='DIR', default="00003")
        parser.add_argument('--network_pkl_stylegan', help='', metavar='DIR', default="network-snapshot-005320.pkl")
        # Inversion options.
        parser.add_argument('--dataset_w_name', help='', metavar='DIR', default="Pelvis_2.1_repo_no_mask-num-375_train-0.70_val-0.20_test-0.10-expinv_00001")
        parser.add_argument('--exp_inv', help='', metavar='DIR', default="00001")
        parser.add_argument('--network_pkl_inv', help='', metavar='DIR', default="")

        # Augmentation options.
        parser.add_argument('--truncation_psi', help='Truncation value.', type=float, default=1.0)
        parser.add_argument('--rand_aug', action='store_true', help='Compute only random GAN augmentation.')
        parser.add_argument('--lower_bound_clip', action='store_true', help='Clip the pixels values under -1 to -1.')
        parser.add_argument('--step_img', help='Selection step to create the image dataset from which compute the distances.', type=int, default=20)
        parser.add_argument('--step_w', help='Selection step to create the latent dataset from which compute the distances.',  type=int, default=5)
        parser.add_argument('--lpips_script', help='How to extract the features manifold.', type=str, default='lpips_script') # 'lpips_script'
        parser.add_argument('--opt_num_epochs', help='Number of optimization steps', type=int, default=10)
        parser.add_argument('--opt_lr', help='Learning rate of optimization algorithm', type=float, default=0.01)
        parser.add_argument('--init_w', help='Initialization point for latent codes [inv | random]', type=str, default='random')

        parser.add_argument('--crop_size_aug', help='Size of the crop applied to images.', type=int, default=64) # 64
        parser.add_argument('--preprocess_aug', help='Type of preprocessing applied for augmentation pipeline [center_crop | random_crop | center_random_crop | original ]', type=str, default='center_random_crop')

        parser.add_argument('--w_pix', help='Weight of recontruction loss', type=float, default=1.0)
        parser.add_argument('--w_lpips', help='Weight of lpips loss', type=float, default=1.0)
        parser.add_argument('--w_latent', help='Weight of latent loss', type=float, default=1.0)
        parser.add_argument('--w_disc', help='Weight of discriminator loss.', type=float, default=1.0)

        parser.add_argument('--p_thres', help='Augmentation probability.', type=float, default=1.0)
        parser.add_argument('--soft_aug', help='Activate smooth augmentation via interpolation.', type=bool, default=False)
        parser.add_argument('--alpha', help='Value for linear interpolation in soft_aug.', type=float, default=1.0) # 0.7
        parser.add_argument('--verbose_log', help='Print losses and time during the optimization process.', type=bool,  default=False)

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseAugment.__init__(self, opt)

        self.gpu_ids_aug = set_gpu_ids(opt.gpu_ids_aug)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids_aug[0])) if self.gpu_ids_aug else torch.device('cpu')  # overwrite device in the base class
        self.phase = opt.phase
        self.batch_size = opt.batch_size

        self.rand_aug = opt.rand_aug
        self.lower_bound_clip = opt.lower_bound_clip
        self.p_thres = opt.p_thres
        self.init_w = opt.init_w # sampling from random point in latent space.

        self.verbose_log = opt.verbose_log
        self.stats_time = []

        if self.phase == 'train':
            print('')
            print('Train phase.')

            if self.rand_aug:
                print('Random GAN augmentation! Disable latent aug parameters.')

                opt.w_pix = 0.0
                opt.w_lpips = 0.0
                opt.w_latent = 0.0
                opt.w_disc = 0.0

                opt.init_w = 'random'
                self.init_w = opt.init_w
                opt.opt_num_epochs = 0
                opt.soft_aug = False

            if self.lower_bound_clip:
                print('Clip pixel values under -1 to -1.')

            self.latent_aug = util_latent_aug.define_latentaugment(
                module_name='latent_aug', phase=opt.phase, opt=opt,  save_dir=self.save_dir, gpu_ids=self.gpu_ids_aug,
            )
            # Addition parameters from the self.opt_augment class.
            self.stats_dataset_w = self.latent_aug.module.stats_dataset_w
            self.num_ws = self.latent_aug.module.num_ws
            self.w_dim = self.latent_aug.module.w_dim
            self.z_dim = self.latent_aug.module.z_dim
        elif self.phase in ['val', 'test']:
            print('')
            print('Val/Test phase.')
            print('All augmentation disabled.')
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def input_sanity_check(img):
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.shape == (1, 256, 256)

    @staticmethod
    def output_sanity_check(img):
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.shape == (1, 256, 256)

    def set_input(self, data):
        assert data['A_paths'] == data['B_paths']

        self.real_A = data['A']
        self.real_B = data['B']

        self.fname = data['A_paths']

        # Merge two batch along channels.
        self.real_AB = torch.cat((self.real_A, self.real_B), dim=1)

    def get_output(self):

        real_AB_aug = self.real_AB_aug.detach().cpu()

        real_A_aug = real_AB_aug[:, 0, :, :].unsqueeze(dim=1) # CT
        real_B_aug = real_AB_aug[:, 1, :, :].unsqueeze(dim=1) # MRI

        # lower bound hard clip.
        if self.lower_bound_clip:
            if real_A_aug.min().item() < -1:
                real_A_aug = torch.clamp(real_A_aug, min=-1.0, max=None)
            if real_B_aug.min().item() < -1:
                real_B_aug = torch.clamp(real_B_aug, min=-1.0, max=None)

        A_path = self.fname
        B_path = self.fname

        data =  {
            'A': real_A_aug, 'B': real_B_aug, 'A_paths': A_path, 'B_paths': B_path
        }

        return data

    def get_latent_output(self):

        w_aug = reverse_broadcasting(self.w_AB_aug)
        w_aug = w_aug.detach().cpu().numpy()
        w_aug = w_aug.squeeze()

        if not self.rand_aug:
            data =  {
                'w': w_aug, 'paths': self.fname,
            }
        else:
            data = {
                'w': w_aug, 'paths': '',
            }

        return data

    def get_latent_input(self):

        w = self.w_AB.detach().cpu().numpy()
        w = w.squeeze()

        if not self.rand_aug:
            data = {
                'w': w, 'paths': self.fname,
            }
        else:
            data = {
                'w': w, 'paths': '',
            }
        return data

    def forward(self): # build around latent_aug() for image-to-image translation task

        # Perform transformations with a probability equal to p_thres.
        since = time.time()
        if random.random() > self.p_thres and self.phase == 'train': # returns a random float number between 0.0 to 1.0
            # Random GAN augmentation.
            if self.rand_aug:
                #if self.dummy_rand_aug:

                w_AB = self.sample_from_randn()
                w_AB = w_AB.to(self.device)
                # Augment.
                self.real_AB_aug, self.w_AB_aug = self.latent_aug.module.forward_ganrand(w_AB)  # random gan augmentation
                self.w_AB = self.w_AB_aug # this type of augmentation keep the sampled latent space to be the same.
            # Latent augmentation.
            else:
                if self.init_w == 'random':
                    # Sample from random.
                    raise NotImplementedError
                elif self.init_w == 'inv':
                    # Sample from inversion.
                    self.w_AB = self.sample_from_inversion(self.fname) # [batch_size, 1, dim_w]
                else:
                    raise NotImplementedError
                # Augment.
                self.w_AB = self.w_AB.to(self.device)
                self.real_AB_aug, self.w_AB_aug = self.latent_aug(self.w_AB, self.fname)
            # Time mantainance.
            time_elapsed = time.time() - since
            if self.verbose_log:
                print('Augmentation completed in {:.0f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))
        else:
            self.real_AB_aug = torch.cat((self.real_A, self.real_B), dim=1)
            # Time mantainance.
            time_elapsed = time.time() - since

            if self.verbose_log:
                print('No augmentation, time {:.0f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))

        self.stats_time.append(time_elapsed)

    #  ------------------------------------------------------------------------------------------------------------------
    # Print Functions

    def sanity_check(self):

        # Sanity check and visualization.
        fname = self.fname[0]
        real_A = self.real_A[0, :, :, :]
        real_B = self.real_B[0, :, :, :]
        self.input_sanity_check(real_A)
        self.input_sanity_check(real_B)
        visualize(real_A, real_B, util_path.get_filename_without_extension(fname), self.save_dir)

        # Perform transformations.
        self.forward()
        data = self.get_output()

        # Sanity check and visualization.
        real_A_aug = data['A'][0, :, :, :]
        real_B_aug = data['B'][0, :, :, :]
        fname_aug = data['A_paths'][0]
        self.output_sanity_check(real_A_aug)
        self.output_sanity_check(real_B_aug)
        visualize(real_A_aug, real_B_aug, util_path.get_filename_without_extension(fname_aug)+'aug', self.save_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # Sampling Functions

    def sample_from_randn(self):
        z = torch.randn([self.batch_size, self.z_dim])
        return z

    def sample_from_inversion(self, fname):
        # Latent code W.
        w = torch.empty([self.batch_size, self.num_ws, self.w_dim], dtype=torch.float32)

        for i, fn in enumerate(fname):
            with self.stats_dataset_w.open_file(fn) as f:
                out_w = pickle.load(f)

            w_fn = torch.from_numpy(out_w.astype("float32")).unsqueeze(dim=0)
            w[i, :, :] = w_fn

        w = reverse_broadcasting(w)
        assert w.shape == (self.batch_size, 1, self.w_dim)

        return w

# Print to screen.
def visualize(imgA, imgB, img_name, save_dir):
    if isinstance(imgA, torch.Tensor):
        imgA = imgA.detach().cpu().numpy()
    if isinstance(imgB, torch.Tensor):
        imgB = imgB.detach().cpu().numpy()
    if len(imgA.shape) == 2:
        img = np.concatenate([imgA, imgB], axis=1)
    else:
        img = np.concatenate([imgA[0, :, :], imgB[0, :, :]], axis=1)
    fig, ax = plt.subplots(figsize=plt.figaspect(img))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img, cmap='gray')
    plt.axis('off')
    fig.savefig(os.path.join(save_dir, f"{img_name}.png"), dpi=400, format='png')
    plt.show()