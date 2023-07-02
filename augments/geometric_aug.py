import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from .base_aug import BaseAugment
from genlib.utils import util_general

import kornia.augmentation as K

class GeometricAugment(BaseAugment):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            parser:   -- original option parser
        Returns:
            the modified parser.

        """
        parser.add_argument('--p_thres', type=float,  default=0.5, help='Augmentation probability.')
        parser.add_argument('--horizontal_flip', action='store_true', help='If specified, flip the images for augmentation')
        parser.add_argument('--affine', action='store_true', help='If specified, rotate|shift|scale images for augmentation')
        parser.add_argument('--elastic_deform', action='store_true', help='If specified, elastic deform the images for augmentation')
        parser.add_argument('--rotate_limit', type=float, default=3, help='Rotation range (-rotate_limit, rotate_limit) in [DEGREE]')
        parser.add_argument('--shift_limit', type=float, default=0.05, help='Shift as a fraction of the image height/width (-shift_limit, shift_limit)')
        parser.add_argument('--verbose_log', help='Print losses and time during the optimization process.', type=bool,  default=False)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseAugment.__init__(self, opt)
        self.p_thres = opt.p_thres

        self.horizontal_flip = opt.horizontal_flip
        self.affine = opt.affine
        self.elastic_deform = opt.elastic_deform
        self.rotate_limit = opt.rotate_limit
        self.shift_limit = opt.shift_limit

        self.verbose_log = opt.verbose_log
        self.stats_time = []

        if opt.phase == 'train':
            print('')
            print('Train phase.')
            self.transform =  self.get_train_transform()
        elif self.phase in ['val', 'test']:
            print('')
            print('Val/Test phase.')
            print('All augmentation disabled.')
        else:
            raise NotImplementedError

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
        self.real_A = data['A'].to(self.device)
        self.real_B = data['B'].to(self.device)

        assert data['A_paths'] == data['B_paths']
        self.fname = data['A_paths']

        # Merge two batch along channels.
        self.real_AB = torch.cat((self.real_A, self.real_B), dim=1)
        self.real_AB = self.real_AB.to(self.device)

    def get_output(self):

        real_AB_aug = self.real_AB_aug.detach().cpu()
        real_A_aug = real_AB_aug[:, 0, :, :].unsqueeze(dim=1)
        real_B_aug = real_AB_aug[:, 1, :, :].unsqueeze(dim=1)
        A_path = self.fname
        B_path = self.fname

        data =  {
            'A': real_A_aug, 'B': real_B_aug, 'A_paths': A_path, 'B_paths': B_path
        }

        return data


    def get_train_transform(self):
        # https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomElasticTransform
        # https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomAffine
        transform_list = []

        # Note the images are already in [-1 1], normalize in [0 1]
        # transform_list += [K.Normalize(mean=(-1,), std=(2,), p=1.0)]

        if self.horizontal_flip:
            print('Horizontal flip ON')
            transform_list += [
            K.RandomHorizontalFlip(p=(1-self.p_thres)) # 0.5
            ]
        if self.affine:
            print('Affine ON')
            transform_list += [
                K.RandomAffine(p=(1-self.p_thres), degrees=self.rotate_limit, translate=self.shift_limit, padding_mode="reflection") # 0.8
            ]
        if self.elastic_deform:
            print('Elastic deform ON')
            transform_list += [
                K.RandomElasticTransform(p=(1-self.p_thres), padding_mode="reflection")
            ]

        # Normalize in [-1, 1].
        # transform_list += [K.Normalize(mean=(0.5,), std=(0.5,), p=1.0)]

        # Compose transformations.
        transform_compose = K.AugmentationSequential(*transform_list)

        return transform_compose

    def get_valid_transform(self):

        transform_list = []
        transform_list += [A.Normalize(mean=(127.5,), std=(127.5,), max_pixel_value=1.0, always_apply=True)]
        transform_list += [ToTensorV2()]

        transform_compose = A.Compose(transform_list, additional_targets={'imageB': 'image'})
        return transform_compose

    def forward(self):
        # Perform transformations.
        since = time.time()
        self.real_AB_aug = self.transform(self.real_AB)
        time_elapsed = time.time() - since
        self.stats_time.append(time_elapsed)

        # Time mantainance.
        if self.verbose_log:
            print('Augmentation completed in {:.0f}m {:.3f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def sanity_check(self):

        # Visualization.
        fname = self.fname[0]
        real_A = self.real_A[0, :, :, :]
        real_B = self.real_B[0, :, :, :]
        self.input_sanity_check(real_A)
        self.input_sanity_check(real_B)
        visualize(
            real_A, real_B, util_general.get_filename_without_extension(fname), self.save_dir
        )
        # Perform transformations.
        self.forward()
        data = self.get_output()

        # Sanity check and visualization.
        real_A_aug = data['A'][0, :, :, :]
        real_B_aug = data['B'][0, :, :, :]
        fname_aug = data['A_paths'][0]
        self.output_sanity_check(real_A_aug)
        self.output_sanity_check(real_B_aug)
        visualize(
            real_A_aug, real_B_aug, util_general.get_filename_without_extension(fname_aug)+'aug', self.save_dir
        )

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




