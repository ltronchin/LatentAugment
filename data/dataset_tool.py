"""Tool for creating ZIP/PNG based datasets."""

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import click

import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image

import yaml
from src.models.stylegan3.genlib.utils import util_general
import json
import glob
import dicom2nifti
from itertools import repeat
import nibabel as nib
from multiprocessing import Pool
import shutil
import random
import scipy
import nilearn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import src.engine.utils.path_utils as path_utils
import src.engine.utils.utils as utils
import src.engine.utils.io_utils as io_utils

from src.models.stylegan3.genlib.utils import util_medical_data
#----------------------------------------------------------------------------

def sanity_check_fllename(data_dir):
    # Patient filename desiderata example "Pelvis_2857,MR Pelvis_2857,MR"
    def count_chars(s):
        result = 0
        for _ in s:
            result += 1
        return result

    def find_chars(s, ch):
        lst = []
        for pos,char in enumerate(s):
            if char == ch:
                lst.append(pos)
        return lst

    # Sanity check
    list_patients = np.asarray(os.listdir(data_dir))
    mask = np.where(np.asarray([count_chars(x.split(',')[-1]) > 4 for x in list_patients], dtype=np.uint8) == 1 )
    list_patients_typo = list_patients[mask]

    print(f"Finded {list_patients_typo.shape[0]} patients with typo in the file name.")
    list_patient_fixed = []
    for patient_typo in list_patients_typo:
        indexes = find_chars(patient_typo, '_')
        patient_fixed_temp= patient_typo[:indexes[1]] + ',' + patient_typo[indexes[1]+1:]
        patient_fixed = patient_fixed_temp[:indexes[-1]] + ',' + patient_fixed_temp[indexes[-1] + 1:]
        print(f"{patient_typo} ----> {patient_fixed}")
        list_patient_fixed.append(patient_fixed)
        os.rename(os.path.join(data_dir, patient_typo), os.path.join(data_dir, patient_fixed))

    # Double check
    mask = np.where(np.asarray([count_chars(x.split(',')[-1]) > 4 for x in os.listdir(data_dir)], dtype=np.uint8) ==1)[0]
    if len(mask) != 0:
        print("Patient with typo in the name after cleaning")
        print(list_patient_fixed[mask])
    else:
        print("Sanity check completed")

def export_statistics(dataset_name, data_dir, reports_dir):

    column_names = ["filename", "id_patient", "sex", "cancer_type", "additional_info"]
    df = pd.DataFrame(columns = column_names)

    list_patients = np.asarray(os.listdir(data_dir))
    for x in list_patients:
        filename = x
        id_patient = x.split(',')[0].split('_')[-1]

        sex = x.split(',')[-1][0]
        cancer_type = x.split(',')[-1][1]
        try:
            addinfo = x.split(',')[-1][2:]
        except IndexError:
            addinfo = ''

        df.loc[len(df.index)] = [filename, id_patient, sex, cancer_type, addinfo]
    df.to_excel(os.path.join(reports_dir,f'info_{dataset_name}.xlsx'), index = False)

#----------------------------------------------------------------------------
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def open_image_folder_patients(source_dir, transpose_img):
    input_patients = glob.glob(os.path.join(source_dir, "*"))

    # Load labels.
    labels = {}

    max_idx = len(input_patients) # maybe_min(len(input_patients), max_patients)

    def iterate_images():
        for idx, patient in enumerate(input_patients):
            arch_fname = os.path.relpath(patient, source_dir)
            arch_fname = arch_fname.replace("\\", "/")

            # Read all modalities of the selected patient.
            modalities = glob.glob(os.path.join(patient, "*.nii.gz"))
            name_modalities = [path_utils.get_filename(path) for path in modalities]
            name_modalities = [name_modalities[i].replace(".nii.gz", "") for i in range(len(name_modalities))] # Remove the .nii.gz

            parent_dir = path_utils.get_parent_dir(modalities[0])
            patient_name = path_utils.get_filename(parent_dir)

            fdata = {}
            for name_modality in name_modalities:
                fdata[name_modality] = nib.load(os.path.join(parent_dir, f"{name_modality}.nii.gz")).get_fdata()

            depth = nib.load(modalities[0]).shape[-1] # Save the number of Slices/Channel
            for d in range(depth):
                img = {}  # dictionary of modalities.
                for name_modality in name_modalities:
                    img[name_modality] = fdata[name_modality][:, :, d]  # based on depth index (d): 0000->0128
                    if transpose_img: # fix orientation
                        img[name_modality] = np.transpose(img[name_modality], [1, 0])
                yield dict(img=img, label=labels.get(arch_fname), name=f"{patient_name:s}_{d:05d}", folder_name=f"{patient_name:s}", depth_index=d, total_depth=depth)

            if idx >= max_idx - 1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_dataset_patient(source, transpose_img):
    if os.path.isdir(source):
        return open_image_folder_patients(source, transpose_img)
    elif os.path.isfile(source):
        assert False, "unknown archive type"
    else:
        error(f"Missing input file or directory: {source}")

#----------------------------------------------------------------------------

def convert_dicom_2_nifti(source: str, dest: str, modes_to_preprocess: list):
    """Function to merge the slices of each patient (dicom) to nifti volume (do ir for each modality)"""
    patients = glob.glob(os.path.join(source, "*")) # patient folders

    # Cycle on patients
    for pat in patients:
        fname_pat = path_utils.get_filename_without_extension(pat)
        print(f"\nPatient: {pat}")
        output_dir = os.path.join(dest, fname_pat)
        if os.path.exists(output_dir):
            print(f"{output_dir} already exist! Skip this patient.")
            continue

        path_utils.make_dir(output_dir, is_printing=False)

        # Cycle on modalities
        for fname_mode in modes_to_preprocess:
            print(f"Modality: {fname_mode}")
            if os.listdir(pat) != fname_mode:
                mode = os.path.join(pat, os.listdir(pat)[0], fname_mode)
            else:
                mode = os.path.join(pat, fname_mode)

            output_file = os.path.join(output_dir, f"{fname_mode}.nii.gz")

            try:
                # reorient_nifti=False to mantain the orientation in Dicom file
                dicom2nifti.dicom_series_to_nifti(mode, output_file, reorient_nifti=False) # to dicom_series_to_nifti feed a directory containing the slices to be included in .nii.gz volume
            except:
                print(f"Fail to convert {mode:s}")

#----------------------------------------------------------------------------

def resize_file(folder_index, folders, dest, image_shape, interpolation='linear'):  # resize 1 patient folder [MR, CT, ...]
    """Process 1 patient"""

    # The folder variable represent the patient.
    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)

    files = glob.glob(os.path.join(folder, "*"))

    # Get all modalities for the current patient.
    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], "*"))

    # For each modality.
    for file_mode in files:
        output_dir = os.path.join(dest, name)
        path_utils.make_dir(output_dir, is_printing=False)

        fname = path_utils.get_filename_without_extension(file_mode)
        output_file = os.path.join(output_dir, f"{fname}.gz")

        try:
            # Read, resample and resize the volume
            # Example:
            # -----
            # If we want to resize from [512 x 512 x d] to [256 256 x d] and the pixelspacing in the source resolution
            # is [1 1 3] we have to perform a respacing operation to [1/(512/256) 1/(512/256) 3]
            # -----
            image = utils.read_image(file_mode, image_shape=image_shape, interpolation=interpolation, crop=None)
            nib.save(image, output_file)
        except:
            raise IOError(f"fail to convert {file_mode:s}")

def resize_nifti_folder(source: str, dest: str, image_shape=(256, 256)):  # rescale from [512, 512, n_slices_patient] -> [res, res, n_slices_patient]

    folders = glob.glob(os.path.join(source, "*"))
    try:
        pool = Pool() # Pool(processes=1)  # Multithreading
        l = pool.starmap(resize_file, zip(range(len(folders)), repeat(folders), repeat(dest), repeat(image_shape)))
        pool.close()
    except:
        for idx_pat in range(len(folders)):
            resize_file(folder_index=idx_pat, folders=folders, dest=dest, image_shape=image_shape)

#----------------------------------------------------------------------------

def get_normalization_range(data, data_options):

    # Upper value
    if data_options['upper_percentile'] is not None:
        upper = np.percentile(data, data_options['upper_percentile'])
    elif data_options['range']['max'] is not None:
        upper = data_options['range']['max']
    else:
        upper = data.max()

    # Lower value
    if data_options['lower_percentile'] is not None:
        lower = np.percentile(data, data_options['lower_percentile'])
    elif data_options['range']['min'] is not None:
        lower = data_options['range']['min']
    else:
        lower = data.min()

    return upper, lower

def normalize_per_dataset(data, dataset, modes_args, low=0.0, hi=255.0):
    """follow this https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py"""
    if dataset == "Pelvis_2.1":
        upper, lower = get_normalization_range(data, modes_args)
        data = np.clip(data, lower, upper)
        data = (data - lower) / (upper - lower) # map between 0 and 1
        data = data * 255 # put between 0 and 255
    elif dataset == 'claro':
        pass
    else:
        raise NotImplementedError(f"Normalization for {dataset} was not implemented.")
    return data

def normalize_file(folder_index, folders, dest, dataset, modes_args):
    """Process 1 patient"""

    # The folder variable represent the patient.
    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)
    files = glob.glob(os.path.join(folder, "*"))

    # Get all modalitites for the current patient.
    if os.path.isdir(files[0]):
        files = glob.glob(os.path.join(files[0], "*"))

    # For each modality.
    for file_mode in files:
        output_dir = os.path.join(dest, name)
        path_utils.make_dir(output_dir, is_printing=False)  # Make directory for each patient if not exist.

        fname = path_utils.get_filename_without_extension(file_mode)
        output_file = os.path.join(output_dir, f"{fname}.gz")

        try:
            print(f"Reading: {file_mode}")
            volume = nib.load(file_mode) # read .nii.gz data
            data = volume.get_fdata() # get pixel data

            data = normalize_per_dataset(data, dataset, modes_args[fname.split('.')[0]])

            affine = volume.affine # take the orientation
            image = nib.Nifti1Image(data, affine=affine)
            nib.save(image, output_file)

        except:
            raise IOError(f"fail to normalize {file_mode:s}")

def normalize_folder(source: str, dest: str,  dataset: str,  modes_args: dict):
    folders = glob.glob(os.path.join(source, "*"))

    try:
       pool = Pool() # Pool(processes=1) # Multithreading
       l = pool.starmap(normalize_file, zip(range(len(folders)), repeat(folders), repeat(dest), repeat(dataset)))
       pool.close()
    except:
        for idx_pat in range(len(folders)):
            normalize_file(folder_index=idx_pat, folders=folders, dest=dest, dataset=dataset, modes_args=modes_args)

# ----------------------------------------------------------------------------
# Mask nifti file

def find_mask_file(folder_index, folders, dest, dataset):
    def get_ref_file(dataset): # get reference modality to mask
        if dataset == "brats20":
            return "t2"
        elif dataset == "spleen":
            return "img"
        elif dataset == "heart":
            return "img"
        elif dataset == "kits19":
            return "imaging"
        elif dataset == "Pelvis_2.1":
            return "MR_nonrigid_CT" # "MR_MR_T2" # "MR_nonrigid_CT" #"MR_MR_T2"
        else:
            raise NotImplementedError(f"{dataset:s} is not implemented")

    def get_largest_connected_region(data):
        # Find the largest connected region.
        if np.sum(data) > 0:
            label, num_label = scipy.ndimage.label(data == 1)
            size = np.bincount(label.ravel())
            biggest_label = size[1:].argmax() + 1
            clump_mask = (label == biggest_label).astype(np.uint8)
            return clump_mask
        else:
            return data

    def remove_small_regions(data, min_size):
        # Let us create a binary mask.
        # It is 0 everywhere `segmentation_mask` is 0 and 1 everywhere else.
        binary_mask = data.copy()
        binary_mask[binary_mask != 0] = 1

        # Now, we perform region labelling. This way, every connected component
        # will have their own colour value.
        labelled_mask, num_labels = scipy.ndimage.label(binary_mask)

        # Let us now remove all the too small regions.
        refined_mask = data.copy()
        minimum_cc_sum = min_size
        for label in range(num_labels):
            if np.sum(refined_mask[labelled_mask == label]) < minimum_cc_sum:
                refined_mask[labelled_mask == label] = 0

        return refined_mask

    def smooth_mask(volume_ref, dataset):
        if dataset == "spleen":
            threshold = 30
            m = (volume_ref.get_fdata() >= threshold).astype(np.uint8)
            m = get_largest_connected_region(m)
            m = scipy.ndimage.morphology.binary_fill_holes(m).astype(np.uint8)
            clump_mask_before = m.copy()
            m = scipy.ndimage.morphology.binary_closing(
                m, structure=np.ones((3, 3, 3))
            ).astype(np.uint8)
            m = m + clump_mask_before
            m = (m > 0).astype(np.uint8)
            m = scipy.ndimage.filters.median_filter(m, size=3)
        elif dataset == "heart":
            threshold = 5
            m = (volume_ref.get_fdata() >= threshold).astype(np.uint8)
            for j in range(m.shape[-1]):
                m[:, :, j] = scipy.ndimage.morphology.binary_closing(
                    m[:, :, j], structure=np.ones((5, 5))
                ).astype(np.uint8)
                m[:, :, j] = get_largest_connected_region(m[:, :, j])
                m[:, :, j] = scipy.ndimage.morphology.binary_fill_holes(m[:, :, j])
                m[:, :, j] = scipy.ndimage.filters.median_filter(m[:, :, j], size=11)
        elif dataset == "kits19":
            threshold = 5
            m = (volume_ref.get_fdata() >= threshold).astype(np.uint8)
            for j in range(m.shape[-1]):
                # m[:, :, j] = scipy.ndimage.morphology.binary_closing(
                #     m[:, :, j], structure=np.ones((5, 5))
                # ).astype(np.uint8)
                # m[:, :, j] = get_largest_connected_region(m[:, :, j])
                m[:, :, j] = scipy.ndimage.morphology.binary_fill_holes(m[:, :, j])
                m[:, :, j] = scipy.ndimage.filters.median_filter(m[:, :, j], size=9)
            m = remove_small_regions(m, min_size=125)

        affine = volume_ref.affine
        mask = nib.Nifti1Image(m, affine=affine)

        return mask

    folder = folders[folder_index]
    name = path_utils.get_filename_without_extension(folder)
    ref = get_ref_file(dataset)
    file_ref = os.path.join(folder, f"{ref}.nii.gz")

    output_dir = os.path.join(dest, name)
    path_utils.make_dir(output_dir)
    output_file = os.path.join(output_dir, f"mask.nii.gz")

    sanity_checkpoint_list = [10, 30, 65, 90, 120]
    [path_utils.make_dir(os.path.join(dest, 'sanity_check', f"{x}"), is_printing=False) for x in sanity_checkpoint_list]

    # try:
    print(f"Reading: {file_ref}")
    volume_ref = nib.load(file_ref)
    if dataset in ["spleen", "heart", "kits19"]:
        mask = smooth_mask(volume_ref, dataset)
    elif dataset == "Pelvis_2.1":
        m = nilearn.masking.compute_epi_mask(volume_ref)  # return a binary nifti object
        m = (m.get_fdata()).astype(np.uint8)  # get mask data
        for j in range(m.shape[-1]):
            m[:, :, j] = scipy.ndimage.morphology.binary_fill_holes(m[:, :, j]).astype(np.uint8)
            fig = plt.figure()
            plt.imshow(m[:, :, j] * 255, cmap='gray')
            plt.axis('off')
            fig.savefig(os.path.join(output_dir, f"mask_{j}.png"))
            plt.close()
            # plt.show()

            if j in sanity_checkpoint_list:
                fig = plt.figure()
                plt.imshow(m[:, :, j] * 255, cmap='gray')
                plt.axis('off')
                fig.savefig(os.path.join(dest, 'sanity_check', f"{j}", f"{name}_mask_{j}.png"))
                plt.close()
                #plt.show()


        affine = volume_ref.affine
        mask = nib.Nifti1Image(m, affine=affine)
    else:
        mask = nilearn.masking.compute_epi_mask(volume_ref)

    nib.save(mask, output_file)

    # except:
    #     raise IOError(f"fail to find mask {file_ref:s}")

def find_mask_folder(source: str, dest: str, dataset: str):
    # multithreading
    folders = glob.glob(os.path.join(source, "*"))

    # find_mask_file(0, folders, dest, dataset)

    pool = Pool() # Pool(processes=1)
    l = pool.starmap(
        find_mask_file,
        zip(range(len(folders)), repeat(folders), repeat(dest), repeat(dataset)),
    )
    pool.close()

#----------------------------------------------------------------------------

def visualize(img):
    x = []
    for k, v in img.items():
        x.append(v)

    x = np.concatenate(x, axis=1)

    import matplotlib.pyplot as plt

    plt.imshow(x, cmap="gray", vmin=0, vmax=255)
    plt.show()

def sanity_check(image, pop_range, dest, patient_folder, mask=None):
    # Sanity check

    x = image['img']

    root_sanity_check_dir = os.path.join(dest, 'sanity_check', patient_folder)
    for m in list(x.keys()):
        sanity_check_dir = os.path.join(root_sanity_check_dir, m)
        path_utils.make_dir(sanity_check_dir, is_printing=False)
        im = Image.fromarray(x[m] / 255)
        im.save(os.path.join(sanity_check_dir, f"{image['name']}.tif"), 'tiff', compress_level=0, optimize=False)
    if mask is not None:
        sanity_check_dir = os.path.join(root_sanity_check_dir, 'mask')
        path_utils.make_dir(sanity_check_dir, is_printing=False)
        im_mask = Image.fromarray(mask*255)
        im_mask.save(os.path.join(sanity_check_dir, f"mask_{image['name']}.tif"), 'tiff', compress_level=0, optimize=False)

    if image['depth_index'] == pop_range:
        ftoprint = 'first_from_stack'
    elif image['depth_index'] == (image['total_depth'] - pop_range - 1):
        ftoprint = 'last_from_stack'
    elif image['depth_index'] ==  image['total_depth']//2:
        ftoprint = 'mid_from_stack'
    else:
        ftoprint = None

    if ftoprint is not None:
        print(f"{ftoprint}: {image['name']}")
        for m in list(x.keys()):
            sanity_check_dir = os.path.join(dest, 'sanity_check', f'{ftoprint}_{pop_range}', m)
            path_utils.make_dir(sanity_check_dir, is_printing=False)
            im = Image.fromarray(x[m] / 255)
            im.save(os.path.join(sanity_check_dir, f"{image['name']}.tif"), 'tiff', compress_level=0, optimize=False)

def  convert_dataset_mi(
        source: str,
        source_mask: str,
        dest: str,
        pop_range: int = 10,
        transpose_img: bool = True,
        apply_mask: bool = True,
        hi: float =255.0,
        low: float =0.0,
        is_overwrite: bool = False,
        is_visualize: bool =False
):

    def fix_orientation(x):
        new_img = {}
        for k, v in x.items():
                # new_img[k] = np.flipud(v.transpose(1, 0))
                new_img[k] = v.transpose(1, 0)
        return new_img

    # The CT registration could be not completed for the first/last slices of the patient stack. "is_idx_to_pos" define the range to skip
    # at the beginning and end of the stack
    if pop_range != 0:
        is_idx_to_pos = np.arange(0, pop_range)
    else:
        is_idx_to_pos = np.array([])
    if os.path.exists(os.path.join(dest, "CT_registration_problem.json")):
        with open(os.path.join(dest, "CT_registration_problem.json")) as f:
            CT_reg_otps = json.load(f)

    # Open normalized folder.
    num_files, input_iter = open_dataset_patient(source, transpose_img) # Core function.
    # Open mask folder
    num_files_mask, input_iter_mask = open_dataset_patient(source_mask, transpose_img)  # Core function.

    # Create a temp folder to be save into zipfile
    temp = os.path.join(dest, "temp")

    if os.path.isdir(temp) and is_overwrite:
        print(f"Removing {temp}")
        shutil.rmtree(temp)
    path_utils.make_dir(temp, is_printing=False)

    dataset_attrs = None

    for (idx, image), (_, image_mask) in zip(enumerate(input_iter), enumerate(input_iter_mask)):

        folder_name = image["folder_name"] # patient name
        idx_name = image["name"]
        archive_fname = f"{folder_name}/{idx_name}.pickle"
        path_utils.make_dir(os.path.join(temp, f"{folder_name}"), is_printing=False)
        out_path = os.path.join(temp, archive_fname)

        # Skip the first/last slices of the stack
        if not is_overwrite and os.path.exists(out_path):
            continue
        if image['depth_index'] in is_idx_to_pos:
            continue
        if image['depth_index'] in (image['total_depth'] - is_idx_to_pos -1):
            continue
        if folder_name in list(CT_reg_otps.keys()):
            if CT_reg_otps[folder_name][0] != -1:
                is_idx_to_pos_temp = np.arange(0, CT_reg_otps[folder_name][0])
            else:
                is_idx_to_pos_temp = np.arange(image['total_depth'], CT_reg_otps[folder_name][1], step=-1)
            if image['depth_index'] in is_idx_to_pos_temp:
                continue

        img = image["img"]

        # Apply mask
        # x1 = np.arange(9.0).reshape((3, 3))
        # x2 =  np.random.choice([0, 1], size=(9,)).reshape((3, 3))
        # np.multiply(x1, x2)
        if apply_mask: # apply mask to both modalities
            img_mask = image_mask["img"]["mask"]
            img_mask = img_mask.astype(np.uint8)

            import matplotlib.pyplot as plt
            sanity_check_dir = os.path.join(dest, 'sanity_check', folder_name, 'check_mask_op')
            path_utils.make_dir(sanity_check_dir, is_printing=False)
            fig = plt.figure()
            gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
            axs = gs.subplots(sharex='col', sharey='row')
            axs[0][0].imshow(img["MR_MR_T2"], cmap='gray')
            axs[0][1].imshow(img["MR_nonrigid_CT"], cmap='gray')
            axs[0][2].imshow(img_mask, cmap='gray')

            for mode in sorted(img.keys()):
                img[mode] = np.multiply(img[mode], img_mask) # Multiply arguments element-wise.
                img[mode] =  img[mode].astype(np.float64)

            axs[1][0].imshow(img["MR_MR_T2"], cmap='gray')
            axs[1][1].imshow(img["MR_nonrigid_CT"], cmap='gray')
            axs[1][2].imshow(img_mask, cmap='gray')
            for ax_row in axs:
                for ax in ax_row:
                    ax.axis('off')
            fig.savefig(os.path.join(sanity_check_dir, f"mask_op_{image['name']}.png"))
            plt.close()
            #plt.show()

        # Fix orientation --> already performed in input_iter
        # img = fix_orientation(img)

        # Transform may drop images.
        if img is None:
            continue

        # Sanity check
        if random.uniform(0, 1) > 0.9:
            if is_visualize:
                visualize(img)
        if apply_mask:
            sanity_check(image=image, pop_range=pop_range, dest=dest, patient_folder=folder_name, mask=img_mask)
        else:
            sanity_check(image=image, pop_range=pop_range, dest=dest, patient_folder=folder_name)

        # Error check to require uniform image attributes across # the whole dataset
        modalities = sorted(img.keys())
        cur_image_attrs = { "width": img[modalities[0]].shape[1], "height": img[modalities[0]].shape[0], "modalities": modalities}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs["width"]
            height = dataset_attrs["height"]
            if width != height:
                error(f"Image dimensions after scale and crop are required to be square.  Got {width}x{height}")
            if width != 2 ** int(np.floor(np.log2(width))):
                error("Image width/height after scale and crop are required to be power-of-two")
        elif dataset_attrs != cur_image_attrs:
            err = [f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}" for k in dataset_attrs.keys()]  # pylint: disable=unsubscriptable-object
            error(f"Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n" + "\n".join(err))

        # Save the dict as a pickle.
        io_utils.write_pickle(img, out_path)

# ----------------------------------------------------------------------------

def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        random.shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def split_list_cross_validation(input_list, n_fold=5, shuffle_list=True, is_test=False):
    if shuffle_list and not is_test:
        random.shuffle(input_list)
    if is_test:
        print(input_list)
    n_valid_sample = round(len(input_list) / 5)
    fold_list = list()
    for i_fold in range(n_fold):
        n_start = i_fold * n_valid_sample
        if i_fold < n_fold - 1:
            n_end = (i_fold + 1) * n_valid_sample
        else:
            n_end = len(input_list)
        val_list = input_list[n_start:n_end]
        if i_fold == 0:
            train_list = [e for i, e in enumerate(input_list) if i >= n_end]
        elif i_fold == n_fold - 1:
            train_list = [e for i, e in enumerate(input_list) if i < n_start]
        else:
            train_list = [e for i, e in enumerate(input_list) if i < n_start] + [
                e for i, e in enumerate(input_list) if i >= n_end
            ]
        fold_list.append([train_list, val_list])
    if is_test:
        print(fold_list)
    return fold_list

def write_to_zip(
        source:                     str,
        dest:                       str = None,
        dataset:                    str = "Pelvis_2.1",
        max_patients:               int = 100000,
        validation_method:          str ='hold_out',
        **kwargs
):

    def add_to_zip(zipObj, patient, split):
        files = glob.glob(os.path.join(patient, "*.pickle"))
        if len(files) == 0:
            files = glob.glob(os.path.join(patient, "*.png"))

        print(f">> Writing {patient} to zip file")
        for file in files:
            filename = os.path.join(
                split,
                path_utils.get_filename_without_extension(patient),
                path_utils.get_filename(file),
            )
            # Add file to zip
            zipObj.write(file, filename)

    # Get all patients in temp folder.
    patients = glob.glob(os.path.join(source, "*[!json]"))
    # Get only the names of patients.
    patients = [path_utils.get_filename_without_extension(patient) for patient in patients]
    assert len(patients) > 0

    # Create a basename depending on <dataset>, <num_patient>, <validation_method>, <validation_split>
    max_patients = min(max_patients, len(patients))
    n_exp = kwargs['n_exp']
    split = kwargs['split']
    train_split, val_split, test_split = split["train"], split["val"], split["test"]
    basename = {
        fold: f"{dataset}-num-{max_patients:d}_val-{validation_method}_exps-{n_exp:d}_fold-{fold:d}_train-{train_split:0.2f}_val-{val_split:0.2f}_test-{test_split:0.2f}" for fold in range(n_exp)
    }

    for fold, fbasename in basename.items():
        # Load split dataset if exist, if not make a new one.
        if dest is None:
            parent_dir = path_utils.get_parent_dir(source)
        else:
            parent_dir = dest
        split_path = os.path.join(parent_dir, "train_val_test_ids", f"{fbasename}.pickle")
        path_utils.make_dir(path_utils.get_parent_dir(split_path), is_printing=False)

        if os.path.exists(split_path):
            print(f"Load saved split, name: {split_path}")
            print('')
            s = io_utils.read_pickle(split_path)
            train_patients, val_patients, test_patients = s["train"], s["val"], s["test"]

        elif validation_method == 'bootstrap':
            # todo implement bootstrap validation in dataset preparation
            raise NotImplementedError(f"{validation_method} is not implemented")

        else: # hold_out
            print(f"Create new split without considering the label distribution. See the script 'create_files.py' otherwise.")
            print('')

            train_split, val_split, test_split = (
                train_split / (train_split + val_split + test_split),
                val_split / (train_split + val_split + test_split),
                test_split / (train_split + val_split + test_split),
            )
            # Shuffle and take max_patients samples from dataset.
            # todo limit the dataset before applying the split
            patients = sorted(patients)
            #random.Random(max_patients).shuffle(
            #    patients)  # comment this line if you don't want to change the order fo the patients across the experiments
            sample_patients = patients[:max_patients]

            train_patients, val_test_patients = split_list(sample_patients, train_split)
            val_patients, test_patients = split_list(val_test_patients, val_split / (val_split + test_split))

            s = {"sample_patients": sample_patients, "train": train_patients, "val": val_patients,  "test": test_patients}
            # Save the training/validation/test split
            with open(os.path.join(parent_dir, "train_val_test_ids", f"{fbasename}.json"), 'w') as f:
                json.dump(s, f, ensure_ascii=False, indent=4)  # save as json
            io_utils.write_pickle(s, split_path)  # save as pickle

        # Init zip file.
        out_path = os.path.join(parent_dir, f"{fbasename}.zip",)

        # Write to zip
        with zipfile.ZipFile(out_path, "w") as zipObj:

            # Write label to zip
            if os.path.exists(os.path.join(source, 'dataset.json')):
                print(f"Zip label: {os.path.join(source, 'dataset.json')}")
                print('')
                zipObj.write(os.path.join(source, 'dataset.json'), 'dataset.json') # source path in folder / path in zip fiile
            for split in ['train', 'valid', 'test']:
                if os.path.exists(os.path.join(source, f"dataset_fold-{fold}_{split}.json")):
                    print(f"Zip label: {os.path.join(source, f'dataset_fold-{fold}_{split}.json')}")
                    print('')
                    zipObj.write(os.path.join(source, f"dataset_fold-{fold}_{split}.json"), f"{split}/dataset.json")

            for patient in train_patients:
                patient_path = os.path.join(source, patient)
                add_to_zip(zipObj, patient_path, "train")
            for patient in val_patients:
                patient_path = os.path.join(source, patient)
                add_to_zip(zipObj, patient_path, "val")
            for patient in test_patients:
                patient_path = os.path.join(source, patient)
                add_to_zip(zipObj, patient_path, "test")



# ----------------------------------------------------------------------------
# CLARO PROCESSING
# ----------------------------------------------------------------------------

def to_pickle():
    fdata = {}
    for name_modality in name_modalities:
        fdata[name_modality] = nib.load(os.path.join(parent_dir, f"{name_modality}.nii.gz")).get_fdata()

    depth = nib.load(modalities[0]).shape[-1]  # Save the number of Slices/Channel
    for d in range(depth):
        img = {}  # dictionary of modalities.
        for name_modality in name_modalities:
            img[name_modality] = fdata[name_modality][:, :, d]  # based on depth index (d): 0000->0128
        yield dict(
            img=img, label=labels.get(arch_fname), name=f"{patient_name:s}_{d:05d}",  folder_name=f"{patient_name:s}", depth_index=d, total_depth=depth
        )

def process_tiff(source:                str,
                 source_interim:        str,
                 source_box:            str,
                 dest:                  str,
                 dataset:               str,
                 resolution:            int,
                 box_value:             str,
                 clip:                  {},
                 scale:                 {},
                 convert_to_uint8:      bool,
                 scale_by_255:          bool,
                 mode:                  list = None,
                 is_overwrite:          bool = True,
                 is_visualize:          bool = False,
                 is_sanity_check:       bool = True,
                 dataset_attrs:         list = None):

    if mode is None:
        mode = ['CT']
    assert len(mode) == 1

    # Create a temp folder.
    temp = os.path.join(dest, "temp")

    if os.path.isdir(temp) and is_overwrite:
        print(f"Removing {temp}")
        shutil.rmtree(temp)
    path_utils.make_dir(temp, is_printing=False)

    # Upload raw_data.
    data_raw = pd.read_csv(os.path.join(source_interim, 'bootstrap', 'folds', 'all.txt'), sep=" ")
    slices = pd.Series([path_utils.get_filename_without_extension(x) for x in data_raw['img']])
    patients = np.unique([path_utils.split_dos_path_into_components(x)[0] for x in data_raw['img']])
    print(f'Number of images: {len(slices)}')
    print(f"Number of patients: {len(patients)}")
    print('Create dataset')
    dataset = util_medical_data.ImgDatasetPreparation(
        data=slices, data_dir=source, resolution=resolution, data_dir_box=source_box, box_value=box_value, clip=clip, scale=scale, convert_to_uint8=convert_to_uint8, scale_by_255=scale_by_255
    )
    print('Initialize dataloader')
    print()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    for x, id_patient, id_slice in dataloader:  # Iterate over the batches of the dataset
        # Save img a modes dict.
        try:
            img = {
                mode[0]: x.detach().cpu().numpy()[0][0]
            }  # dictionary.

            # id_patient.
            id_patient = id_patient[0]
            # id_slice.
            id_slice = id_slice[0]

            # Format path temp/<id_patient>/<id_patient>_<id_slice>
            archive_fname = f"{id_patient}/{id_patient}_{int(id_slice):05d}.pickle"
            print(archive_fname)
            path_utils.make_dir(os.path.join(temp, id_patient), is_printing=False)
            out_path = os.path.join(temp, archive_fname)

            if not is_overwrite and os.path.exists(out_path):
                continue

            # Transform may drop images.
            if img is None:
                print("Discarded image!")
                print(f"Check: {archive_fname}")
                print('')
                break

            # Running visualization.
            if is_visualize:
                if random.uniform(0, 1) > 0.9:

                    plt.imshow(img[mode[0]], cmap="gray", vmin=0, vmax=255)
                    plt.axis('off')
                    plt.show()

            # Sanity check.
            if is_sanity_check:
                sanity_check_dir = os.path.join(dest, 'sanity_check', id_patient)
                path_utils.make_dir(sanity_check_dir, is_printing=False)
                im = Image.fromarray(img[mode[0]] / 255)
                im.save(os.path.join(sanity_check_dir, f"{id_patient}_{int(id_slice):05d}.tif"), 'tiff', compress_level=0, optimize=False)
            modalities = sorted(img.keys())
            cur_image_attrs = {
                "width": img[modalities[0]].shape[1],   "height": img[modalities[0]].shape[0], "modalities": modalities, "dtype": img[modalities[0]].dtype
            }
            if dataset_attrs is None: # if is None, stylegan2-ada preprocessing
                dataset_attrs = cur_image_attrs
                width = dataset_attrs["width"]
                height = dataset_attrs["height"]
                img_dtype = dataset_attrs["dtype"]
                if width != height:
                    error(f"Image dimensions after scale and crop are required to be square.  Got {width}x{height}")
                if width != 2 ** int(np.floor(np.log2(width))):
                    error("Image width/height after scale and crop are required to be power-of-two")
                if img_dtype != 'float64':
                    error("Medical Stylegan2-ada? I want float data!")
            elif dataset_attrs != cur_image_attrs:
                err = [f"  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}" for k in  dataset_attrs.keys()]  # pylint: disable=unsubscriptable-object
                error(f"Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n" + "\n".join(err))

            # Save to pickle.
            io_utils.write_pickle(img, out_path)
        except BaseException as err:  # to consider exception not listed above. Use it with
            print(f"Unexpected {err}, {type(err)}")
            raise  # to raise the error


# ----------------------------------------------------------------------------
@click.command()
@click.option('--seed', help='Name of the input dataset', required=True, type=int, default=42)
@click.option('--configuration_file', help='Path to configuration file', required=True, metavar='PATH')
@click.option('--data_dir', help='Directory for input dataset', required=True, metavar='PATH')
@click.option('--data_dir_mask', help='Directory for mask dataset', metavar='PATH')
@click.option('--data_dir_box', help='Directory for bounding box file', metavar='PATH')

@click.option('--interim_dir', help='Output directory for output dataset', required=True, metavar='PATH')
@click.option('--reports_dir', help='Output directory for reports', required=True, metavar='PATH')
@click.option('--dataset', help='Name of the input dataset', required=True, type=str, default='Pelvis_2.1')
@click.option('--max_patients', help='Number of patients to preprocess', type=int, default=100000)
@click.option('--processing_step', help='Processing step', type=click.Choice(['process_dicom_2_nifti', 'process_nifti_resized', 'process_nifti_normalized', 'mask_nifti', 'snap_pickle', 'snap_zip', 'process_tiff']), required=True)
def main(**kwargs):
    opts = EasyDict(**kwargs)

    # Configuration file
    print("Upload configuration file")
    with open(opts.configuration_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # Submit run:
    print("Submit run")
    run_module = os.path.basename(__file__)
    run_id = util_general.get_next_run_id_local(os.path.join('log_run', opts.dataset), run_module)  # GET run id
    # Create log dir
    run_name = "{0:05d}--{1}--{2}".format(run_id, run_module, opts.processing_step)
    log_dir = os.path.join('log_run', opts.dataset, run_name)
    util_general.create_dir(log_dir)
    # Save the configuration file
    with open(os.path.join(log_dir, 'configuration.json'), 'w') as f:
        json.dump(opts, f, ensure_ascii=False,  indent=4)
    # Initialize Logger
    logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
    # Copy the code in log_dir
    files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store'],  add_base_to_relative=True)
    files = [(f[0], os.path.join(log_dir, f[1])) for f in files]
    util_general.copy_files_and_create_dirs(files)

    # Welcome
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print("Hello!", date_time)

    # Seed everything
    print("Seed all")
    util_general.seed_all(opts.seed)

    # Files and Directories
    print('Create file and directory')
    data_dir = opts.data_dir
    interim_dir = os.path.join(opts.interim_dir, opts.dataset)
    reports_dir = os.path.join(opts.reports_dir,  opts.dataset)
    path_utils.make_dir(reports_dir)

    if 'claro' in opts.dataset:
        data_dir_box = opts.data_dir_box
        prepare_claro(
            data_dir, interim_dir, reports_dir, data_dir_box, cfg=cfg, opts=opts
        )
    elif opts.dataset == 'Pelvis_2.1':
        data_dir_mask = opts.data_dir_mask
        prepare_Pelvis_2_1(
            data_dir, interim_dir, reports_dir, data_dir_mask, cfg=cfg, opts=opts
        )
    else:
        raise NotImplementedError(f"{opts.dataset:s} is not implemented")
    print("May be the force with you!")

def prepare_claro(data_dir, interim_dir, reports_dir, data_dir_box, cfg, opts):

    # Inherit configuration and options file.
    # From config.
    resolution = cfg['data']['resolution']
    modes_args = cfg['data']['modes']
    apply_box =  cfg['data']['options']['apply_box']
    box_value = cfg['data']['options']['box_value']
    transpose_img = cfg['data']['options']['transpose_img']
    convert_to_uint8 = cfg['data']['options']['convert_to_uint8']
    scale_by_255 = cfg['data']['options']['scale_by_255']
    validation_method =  cfg['data']['validation']['name']
    validation_split = cfg['data']['validation']['split']
    n_exp = cfg['data']['validation']['n_exp']
    # From options.
    dataset = opts.dataset
    max_patients = opts.max_patients
    processing_step = opts.processing_step

    # Useful print
    print()
    print('Training options:')
    print()
    print(f'Data directory:      {data_dir}')
    if data_dir_box is not None:
        print(f'Data box directory:  {data_dir_box}')
    print(f'Output directory:    {interim_dir}')
    print(f'Report directory:    {reports_dir}')
    print(f'Dataset resolution:  {resolution}')
    print(f'Modes list:          {list(modes_args.keys())}')
    print(f'Modes args:          {modes_args}')
    print(f'Max patients:        {max_patients}')
    print(f'Processing step:     {processing_step}')
    print(f'Validation method:   {validation_method}')
    print(f'Validation split:    {validation_split}')
    print(f'Number of fold:      {n_exp}')
    print(f'Apply mask:          NOT APPLICABLE')
    print(f'Apply box:           {apply_box}')
    print(f'Box value:           {box_value}')
    print(f'Number of slice to drop and the beginning/end od the stack:    NOT APPLICABLE')
    print(f'Transpose the slice [x y] --> [y x]:                           {transpose_img}')
    print(f'Convert images to uint8:                                       {convert_to_uint8}')
    print(f'Scale form [0.0 1.0] to [0.0 255.0]:                           {scale_by_255}')
    print()

    # Process tiff.
    if processing_step == 'process_tiff':
        dest_dir = interim_dir
        process_tiff(
            source=data_dir, source_interim=interim_dir, source_box=data_dir_box, dest=dest_dir, dataset=dataset, resolution=resolution,  box_value=box_value, clip=modes_args['CT']['clip'], scale=modes_args['CT']['scale'], convert_to_uint8=convert_to_uint8, scale_by_255=scale_by_255
        )

    # Save as zip
    if processing_step == 'snap_zip':
        dest_dir = interim_dir
        print(f"\nSave to zip, output folder: {dest_dir}")
        # Save as zip
        if processing_step == 'snap_zip':
            dest_dir = interim_dir
            print(f"\nSave to zip, output folder: {dest_dir}")
            write_to_zip(
                source=os.path.join(data_dir), dest=dest_dir, dataset=dataset, max_patients=max_patients, validation_method=validation_method, n_exp=n_exp, split=validation_split
            )

def prepare_Pelvis_2_1(data_dir, interim_dir, reports_dir, data_dir_mask, cfg, opts):

    # Inherit configuration and options file.
    # From config.
    resolution = cfg['data']['resolution']
    modes_args = cfg['data']['modes']
    apply_mask =  cfg['data']['options']['apply_mask']
    transpose_img = cfg['data']['options']['transpose_img']
    pop_range = cfg['data']['options']['pop_range']
    validation_method =  cfg['data']['validation']['name']
    validation_split = cfg['data']['validation']['split']
    n_exp = cfg['data']['validation']['n_exp']
    # From options.
    dataset = opts.dataset
    max_patients = opts.max_patients
    processing_step = opts.processing_step

    # Useful print
    print()
    print('Training options:')
    print()
    print(f'Data directory:      {data_dir}')
    if data_dir_mask is not None:
        print(f'Data mask directory:    {data_dir_mask}')
    print(f'Output directory:    {interim_dir}')
    print(f'Report directory:    {reports_dir}')
    print(f'Dataset resolution:  {resolution}')
    print(f'Modes list:          {list(modes_args.keys())}')
    print(f'Modes args:          {modes_args}')
    print(f'Max patients:        {max_patients}')
    print(f'Processing step:     {processing_step}')
    print(f'Validation method:   {validation_method}')
    print(f'Number of fold:      {n_exp}')
    print(f'Validation split:    {validation_split}')
    print(f'Apply mask:          {apply_mask}')
    print(f'Apply box:           NOT APPLICABLE')
    print(f'Box value:           NOT APPLICABLE')
    print(f'Number of slice to drop and the beginning/end od the stack:    {pop_range}')
    print(f'Transpose the slice [x y] --> [y x]:                           {transpose_img}')
    print(f'Convert images to uint8:                                        NOT APPLICABLE')
    print(f'Scale form [0.0 1.0] to [0.0 255.0]:                           True')
    print()

    sanity_check_fllename(data_dir=data_dir)

    path = Path(os.path.join(interim_dir, f'info_{dataset}.xlsx'))
    if path.is_file():
        print('The info file of the selected dataset already exists.')
    else:
        print(f'Export statistics.')
        export_statistics(
            dataset_name=dataset, data_dir=data_dir, reports_dir=interim_dir
        )

    # From dicom to nifti
    if processing_step == 'process_dicom_2_nifti':
        dest_dir = os.path.join(interim_dir, 'nifti_volumes')
        print(f"\nConvert to nifti, output folder: {dest_dir}")
        convert_dicom_2_nifti(
            source=data_dir, dest=dest_dir, modes_to_preprocess=list(modes_args.keys())
        )

    # Resize nifti volume from [original_res, original_res, n_slices] to [res x res x n_slices]
    if processing_step == 'process_nifti_resized':
        dest_dir = os.path.join(interim_dir, f'nifti_volumes_{resolution}x{resolution}')
        print(f"\nResize to resolution {resolution}, output folder: {dest_dir}")
        resize_nifti_folder(
            source=data_dir, dest=dest_dir, image_shape=(resolution, resolution)
        )

    # Normalize each volume
    if processing_step == 'process_nifti_normalized':
        dest_dir = os.path.join(interim_dir, f'nifti_volumes_{resolution}x{resolution}_normalized')
        print(f"\nNormalize each volume, output folder: {dest_dir}")
        normalize_folder(
            source=data_dir, dest=dest_dir, dataset=dataset, modes_args=modes_args
        )

    # Find mask of each volume using CT data
    if processing_step == 'mask_nifti':
        dest_dir = os.path.join(interim_dir, f'nifti_volumes_{resolution}x{resolution}_mask')
        print(f"\nMask each volume using nilearn package, output folder: {dest_dir}")
        find_mask_folder(
            source=data_dir,  dest=dest_dir,  dataset=dataset,
         )

    # Write to pickle
    if processing_step == 'snap_pickle':
        dest_dir = interim_dir
        print(f"\nSave to pickle, output_folder: {dest_dir}")
        print(f"-----")
        print("Each patient folder contains one .pkl file per slice")
        print("Each .pkl file contains all the modes associated to the current slice and the current patient")
        print(f"-----")
        convert_dataset_mi(
            source=data_dir, source_mask=data_dir_mask,  dest=dest_dir, pop_range=pop_range, transpose_img = transpose_img, apply_mask=apply_mask, hi=255.0, low=0.0
        )

    # Save as zip
    if processing_step == 'snap_zip':
        dest_dir = interim_dir
        print(f"\nSave to zip, output folder: {dest_dir}")
        write_to_zip(
            source= os.path.join(data_dir), dest=dest_dir,  dataset=dataset, max_patients=max_patients, validation_method=validation_method, split=validation_split
        )
#----------------------------------------------------------------------------

if __name__ == "__main__":

    main()