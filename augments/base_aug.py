import os
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import random

class BaseAugment(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseAugment.__init__(self, opt).
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseAugment class.
        Parameters:
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
            parser:   -- original option parser
        Returns:
            the modified parser.

        """
        return parser

    @abstractmethod
    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            data (dict): includes the data itself and its metadata information.
        """
        pass


    @abstractmethod
    def forward(self):
        pass


    def get_train_transform(self):
        """
        Parameters:
        """
        pass

    def get_valid_transform(self):
        """
        Parameters:
        """
        pass

    def sanity_check(self):
        pass

def visualize(imgA, imgB):
    pass