import argparse
import os
import torch
import re

# Custom library
import data
import augments
from utils import util_path, util_logger

class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters

        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='pelvis2.1', help='chooses how datasets are loaded.')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--aug', type=str, default=None,  help='Augmentation mode [geometric | latent]')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),  help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        # ...

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify augment-related parser options
        augment_name = opt.aug
        if augment_name is not None:
            aument_option_setter = augments.get_option_setter(augment_name)
            parser = aument_option_setter(parser, self.isTrain)
            opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util_path.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, args=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        if args is not None:
            keys = list(args.keys())
            if 'n_imgs' in keys:
                opt.n_imgs = args['n_imgs']
            if opt.aug == 'geometric':
                if 'p_thres' in keys:
                    opt.p_thres = args['p_thres']
                if 'horizontal_flip' in keys:
                    opt.horizontal_flip = args['horizontal_flip']
                if 'affine' in keys:
                    opt.affine = args['affine']
                if 'elastic_deform' in keys:
                    opt.elastic_deform = args['elastic_deform']
            elif opt.aug == 'latent' and opt.rand_aug:
                if 'p_thres' in keys:
                    opt.p_thres = args['p_thres']
                if 'truncation_psi' in keys:
                    opt.truncation_psi = args['truncation_psi']
            else:
                if 'p_thres' in keys:
                    opt.p_thres = args['p_thres']
                if 'opt_num_epochs' in keys:
                    opt.opt_num_epochs = args['opt_num_epochs']
                if 'opt_lr' in keys:
                    opt.opt_lr = args['opt_lr']
                if 'w_lpips' in keys:
                    opt.w_lpips =  args['w_lpips']
                if 'w_pix' in keys:
                    opt.w_pix = args['w_pix']
                if 'w_latent' in keys:
                    opt.w_latent = args['w_latent']
                if 'w_disc' in keys:
                    opt.w_disc = args['w_disc']
                if 'init_w' in keys:
                    opt.init_w = args['init_w']

        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.aug is not None:
            if opt.aug == 'geometric':
                suffix = f"n_imgs_{opt.n_imgs}-p_thres_{opt.p_thres}-horizontal_flip_{opt.horizontal_flip}-affine_{opt.affine}-elastic_deform_{opt.elastic_deform}"
            elif opt.aug == 'latent' and opt.rand_aug:
                suffix = f"n_imgs_{opt.n_imgs}-truncation_psi_{opt.truncation_psi}"
            else:
                suffix = f"n_imgs_{opt.n_imgs}-opt_lr_{opt.opt_lr}-opt_num_epochs_{opt.opt_num_epochs}-w_latent_{opt.w_latent}-w_pix_{opt.w_pix}-w_lpips_{opt.w_lpips}-w_disc_{opt.w_disc}"
            opt.name = opt.name + '-' + suffix

        #if opt.suffix:
        #    suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #    opt.name = opt.name + suffix

        # Set up the logger
        util_path.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
        util_logger.Logger(file_name=os.path.join(opt.checkpoints_dir, opt.name, 'log.txt'), file_mode='a', should_flush=True)

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
