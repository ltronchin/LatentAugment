"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom augment class called 'dummy', you need to add a file called 'dummy_augment.py' and define a subclass DummyAugment inherited from BaseAugment.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseAugment.__init__(self, opt).
    --
    --
    --
    --

In the function <__init__>, you need to define four lists:
    --
    --
    --
    --

Now you can use the augment class by specifying flag '--augment dummy'.
See our template augment class 'template_augment.py' for more details.
"""

import importlib
from augments.base_aug import BaseAugment

# Custom import
import torch


def find_augment_using_name(augment_name):
    """Import the module "augments/[augment_name]_aug.py".

    In the file, the class called DatasetNameAugment() will
    be instantiated. It has to be a subclass of BaseAugment,
    and it is case-insensitive.
    """
    augment_filename = "augments." + augment_name + "_aug"
    augmentlib = importlib.import_module(augment_filename)
    augment = None
    target_augment_name = augment_name.replace('_', '') + 'augment'
    for name, cls in augmentlib.__dict__.items():
        if name.lower() == target_augment_name.lower() and issubclass(cls, BaseAugment):
            augment = cls

    if augment is None:
        print("In %s.py, there should be a subclass of BaseAugment with class name that matches %s in lowercase." % (
        augment_filename, target_augment_name))
        exit(0)

    return augment


def get_option_setter(augment_name):
    """Return the static method <modify_commandline_options> of the augment class."""
    augment_class = find_augment_using_name(augment_name)
    return augment_class.modify_commandline_options


def create_augment(opt):
    """Create an augment pipeline given the option.

        This function warps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

        Example:
            >>> from augments import create_augment
            >>> augment = create_augment(opt)
        """

    augment = find_augment_using_name(opt.aug)
    instance = augment(opt)

    print("Augment [%s] was created" % type(instance).__name__)
    return instance