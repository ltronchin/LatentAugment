import os
import pickle
import random
import zipfile

import dnnlib
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import util_path


class BatchSampleDatasetCoupled:
    def __init__(self, dataA, dataB):
        self.dataA = dataA
        self.dataB = dataB

    def __getitem__(self, n):
        return self.dataA[n], self.dataB[n]

    def __len__(self):
        return len(self.dataA)

class BatchSampleDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, n):
        return self.data[n]

    def __len__(self):
        return len(self.data)

class DatasetStats:
    def __init__(self, manifold, capture_all=False, max_items=None, step=1):
        self.manifold = manifold
        self.capture_all = capture_all
        self.max_items = max_items
        self.num_items = 0
        self.step = step
        self.all_x = []

        # Define a selection schedule.
        self.schedule = sorted([f'{i:05d}' for i in np.arange(start=10, stop=120 + 1, step=self.step)])
        #print(f'Selected per patient: {[*self.schedule]}')

        # Defining the manifold to work on
        if self.manifold =='latent':
            self.ndim = 3
        elif self.manifold == 'features':
            self.ndim = 4
        elif self.manifold == 'features_jit':
            self.ndim = 2
        elif self.manifold == 'img':
            self.ndim = 4
        else:
            raise NotImplementedError("Unrecognised manifold! Add it!")

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x, fname):

        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == self.ndim

        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return -1
            x = x[:self.max_items - self.num_items]

        if self.capture_all:
            self.all_x.append(x)
            self.num_items += x.shape[0]
            return x.shape[0]
        else:
            *_, idd = util_path.split_dos_path_into_components(path=fname[0])
            idd = util_path.get_filename_without_extension(path=idd)[-5:]
            if idd in self.schedule:
                self.all_x.append(x)
                self.num_items += x.shape[0]
                return x.shape[0]
            return 0

    def append_torch(self, x, idd=None):
        assert isinstance(x, torch.Tensor) and x.ndim == self.ndim
        assert x.shape[0] == 1

        return self.append(x.cpu().numpy(), idd)

    def append_list(self, raw_list, fname=None):
        assert isinstance(raw_list, list)
        assert raw_list[0].shape[0] == 1

        x_list = [x.cpu().numpy() for x in raw_list]
        x_list = [np.asarray(x, dtype=np.float32) for x in x_list]
        x_shape = x_list[0].shape[0]
        x_ndim = x_list[0].ndim
        assert x_ndim == self.ndim

        if (self.max_items is not None) and (self.num_items + x_shape > self.max_items):
            if self.num_items >= self.max_items:
                return -1
            x_list = [x[:self.max_items - self.num_items] for x in x_list]
            x_shape = x_list[0].shape[0]

        if self.capture_all:
            self.all_x.append(x_list)
            self.num_items += x_shape
            return x_shape
        else:
            *_, idd = util_path.split_dos_path_into_components(path=fname[0])
            idd = util_path.get_filename_without_extension(path=idd)[-5:]
            if idd in self.schedule:
                self.all_x.append(x_list)
                self.num_items += x_shape
                return x_shape
            return 0

    def get_all(self):
        return np.concatenate(self.all_x, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all().astype(np.float32))

    def get_all_list(self):
        x_torch_list = []
        for i in range(len(self.all_x[0])):

            x_numpy = np.concatenate([x_list[i] for x_list in self.all_x], axis=0)
            x_torch = torch.from_numpy(x_numpy.astype(np.float32))
            x_torch_list.append(x_torch)

        return x_torch_list

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = DatasetStats(manifold=s.manifold, capture_all=s.capture_all, max_items=s.max_items, step=s.step)
        obj.__dict__.update(s)
        return obj

# Datasets utils.
class LatentCodeDataset(torch.utils.data.Dataset):
    def __init__(self,  path, split, w_dim=512, num_ws=14):

        self._path = path
        self._split = split
        self._zipfile = None

        # Check if the dataset is stored as zip file.
        if self._file_ext(self._path) == ".zip":
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a zip")

        # Extract all the filenames stored in the zip file.
        self._fnames = [fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and self._split in fname]
        self._fnames = sorted(self._fnames)
        if len(self._fnames) == 0:
            raise IOError("No files found in the specified path")

        # Sanity checks.
        raw_shape = [len(self._fnames)] + list(self._load_w(raw_idx=0)[0].shape)
        if w_dim is not None and (raw_shape[2] != w_dim):
            raise IOError("W does not match the specified latent dimension.")
        if num_ws is not None and (raw_shape[1] != num_ws):
            raise IOError("W does not match the specified broadcasting.")

        # Define the list of possible indexes.
        self._raw_shape = list(raw_shape)
        self._raw_idx = np.arange( self._raw_shape[0], dtype=np.int64)

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        w, fname = self._load_w(self._raw_idx[idx])
        return w, fname

    # Zip utilities.
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def open_file(self, fname):
        return self._get_zipfile().open(fname, "r")

    def _load_w(self, raw_idx):
        fname = self._fnames[raw_idx]

        # Latent code W.
        with self.open_file(fname) as f:
            out_w = pickle.load(f)
        out_w = out_w.astype("float32")

        return out_w, fname

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, modalities, resolution=256):

        self._path = path
        self._split = split
        self._modalities = modalities
        self._zipfile = None


        # Check if the dataset is stored as zip file.
        if self._file_ext(self._path) == ".zip":
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError("Path must point to a zip")

        # Extract all the filenames stored in the zip file.
        self._fnames = [
            fname for fname in self._all_fnames if self._file_ext(fname) == ".pickle" and self._split in fname
        ]
        self._fnames = sorted(self._fnames)
        if len(self._fnames) == 0:
            raise IOError("No files found in the specified path")

        # Sanity checks.
        raw_shape = [len(self._fnames)] + list(self._load_raw_image(raw_idx=0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError("Image files do not match the specified resolution")
        if len(modalities) is not None and (raw_shape[1] != len(modalities)):
            raise IOError("Image does not match the specified number of channels.")

        # Define the list of possible indexes.
        self._raw_shape = list(raw_shape)
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        w, fname = self._load_raw_image(self._raw_idx[idx])
        return w, fname

    # Zip utilities.
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        return self._get_zipfile().open(fname, "r")

    def _load_raw_image(self, raw_idx):
        fname = self._fnames[raw_idx]

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

# ----------------------------------------------------------------------------------------------------------------------
# Utils to crop images.

def get_params(load_size, crop_size, preprocess='center_random_crop'):
    w, h = (load_size, load_size)
    new_w = w
    new_h = h

    assert preprocess in ['center_random_crop', 'random_crop']

    if preprocess == 'center_random_crop':
        new_w = new_h = int(np.sqrt((load_size * load_size) / 2))

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))
    return {'crop_pos': (x, y)}

def get_transform(load_size, crop_size, preprocess, params=None):
    transform_list = []

    if preprocess == 'center_crop':
        transform_list.append(transforms.CenterCrop(int(np.sqrt((load_size * load_size) / 2))))
    elif preprocess == 'random_crop':
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: crop(img, params['crop_pos'], crop_size)))
    elif preprocess == 'center_random_crop':
        transform_list.append(transforms.CenterCrop(int(np.sqrt((load_size * load_size) / 2))))
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: crop(img, params['crop_pos'], crop_size)))

    return transforms.Compose(transform_list)

def get_center_crop(load_size):
    transform_list = [
        transforms.CenterCrop(
            int(np.sqrt((load_size * load_size) / 2))
        )
    ]
    return transforms.Compose(transform_list)

def crop(img, pos, size):
    assert isinstance(img, torch.Tensor)
    _, _, ow, oh = img.shape
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img[:, :, y1 : y1 + th,  x1 : x1 + tw] # img.crop((x1, y1, x1 + tw, y1 + th))
    return img