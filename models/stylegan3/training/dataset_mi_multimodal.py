# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

# CUSTOMIZING START
import pickle
#import pickle5 as pickle
# CUSTOMIZING END

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        random_seed,
        # CUSTOMIZING START
        dtype,                  # dtype of the images in the dataset
        split       = "train",  # Dataset to train
        modalities  = None,     # Input modalities for StyleGAN.
        # CUSTOMIZING END
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip        = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        **kwargs
    ):

        self._name = name
        # CUSTOMIZING START
        self._dtype = dtype
        self._split = split
        if modalities is None:
            modalities = ['MR_nonrigid_CT', 'MR_MR_T2']
        self._modalities = modalities
        # CUSTOMIZING END
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image, fname_image = self._load_raw_image(self._raw_idx[idx]) # function from the dataset ImageFolderDataset
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        # CUSTOMIZING START
        assert image.dtype == self._dtype
        # CUSTOMIZING END
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1] # HERE THE x_flip is applied! The images is MIRRORED (left-to-right)
        return image.copy(), self.get_label(idx), fname_image

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        # CUSTOMIZING START
        # d.xflip = (int(self._xflip[idx]) != 0)
        d.xflip = int(self._xflip[idx]) != 0
        # CUSTOMIZING END
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    # CUSTOMIZING START
    @property
    def dtype(self):
        return self._dtype
    @property
    def modatilies(self):
        return self._modalities
    @property
    def split(self):
        return self._split
    @property
    def mean_nslices(self):
        return self._mean_nslices
    # CUSTOMIZING END

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

# CUSTOMIZING START
class CustomImageFolderDataset(Dataset):
    def __init__(
        self,
        path,                # Path to directory or zip.
        resolution  = None,  # Ensure specific resolution, None = highest available.
        get_info    = False,
        perc_size   = 1.0,   # Artificially limit the number of the patient of the dataset.
        split_patients=None,
        random_seed = 0,     # Random seed to use when applying max_size.
        **super_kwargs,      # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._mean_nslices = None
        self._split = super_kwargs["split"]
        self._modalities = super_kwargs["modalities"]

        # Check if the dataset is stored as zip file (Only zip file accepted)
        if self._file_ext(self._path) == ".zip":
            self._type = "zip"
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a directory or zip")

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and self._split in fname)
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        # CUSTOMIZING START
        # Patients split.
        if perc_size < 1.0:
            print(f'Limiting {self._split} patients by {perc_size} dataset.')
            try:
                with open(os.path.join(split_patients, f'train_patients_{perc_size}.txt')) as f:
                    self._p_fnames = f.read()
                self._p_fnames = np.asarray(self._p_fnames.split('\n'))
                self.max_p = self._p_fnames.size
                print('List uploaded.')
            except FileNotFoundError:
                self._p_fnames = np.unique([fname.split(sep='/')[1] for fname in self._image_fnames])
                self.max_p = round(perc_size * self._p_fnames.size)

                if self._p_fnames.size > self.max_p:
                    np.random.RandomState(random_seed).shuffle(self._p_fnames)
                    self._p_fnames = np.sort(self._p_fnames[:self.max_p])
                print('List created.')
            # Filter.
            self._image_fnames = [fname for fname in self._image_fnames if any(fname.split(sep='/')[1] == self._p_fnames)]
        else:
            self._p_fnames = np.unique([fname.split(sep='/')[1] for fname in self._image_fnames])
            self.max_p = self._p_fnames.size
        # CUSTOMIZING STOP

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape) # added [0] to select the image from the list [img, img_fname]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")

        if get_info:
            self._get_info()

        super().__init__(name=name, raw_shape=raw_shape, random_seed=random_seed, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        else:
            raise IOError("Support only zip.")

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    # CUSTOMIZING START
    def _get_info(self):
        assert len(self._image_fnames) != 0
        self._all_patient_fnames = np.unique(
            sorted(os.path.split(img_name)[0] for img_name in self._image_fnames)
        )
        self._all_patient_nslices = {
            pname: len([img_name for img_name in self._image_fnames if pname in img_name]) for pname in  self._all_patient_fnames
        }
        self._mean_nslices = np.mean(list(self._all_patient_nslices.values()))
        print('')
        print(f'Path:                                               {self._path}')
        print(f'Split:                                              {self._split}')
        print(f'Modalities:                                         {self._modalities}')
        print(f'Number of {self._split} patients:                   {len(self._all_patient_fnames)}')
        print(f'Number of {self._split} slices:                     {len(self._image_fnames)}')
        print(f'Mean number of slices for {self._split} patientes:  {self._mean_nslices}')
        print('')
    # CUSTOMIZING STOP

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        with self._open_file(fname) as f:
            p = pickle.load(f)

        assert len(self._modalities) > 0
        s = p[self._modalities[0]]

        out_image = np.zeros((len(self._modalities), s.shape[0], s.shape[1])).astype("float32") # compose the multichannel image
        for i, _modality in enumerate(self._modalities):
            x = p[_modality]
            x = x.astype("float32")
            out_image[i, :, :] = x
        return out_image, fname # CHW

    def _load_raw_labels(self):
        # CUSTOMIZATION START
        fname = f"{self._split}/dataset.json" # fname = "dataset.json"
        # CUSTOMIZATION STOP
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)["labels"]
        if labels is None:
            return None
        labels = dict(labels)
        # CUSTOMIZATION START
        labels = [labels[os.path.relpath(fname.replace("\\", "/"), f"{self._split}/")] for fname in self._image_fnames] # labels = [labels[fname.replace("\\", "/")] for fname in self._image_fnames]
        #print(f'Labels size:        {len(labels)}')
        #print(f'Images size:        {len(self._image_fnames)}')
        assert len(labels) == len(self._image_fnames)
        # CUSTOMIZATION STOP
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
# CUSTOMIZING END

#----------------------------------------------------------------------------
