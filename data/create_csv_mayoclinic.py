import os
import random
import pandas as pd
import pydicom
from tqdm import tqdm
import shutil

def make_split(data, split=0.8, shuffle_list=True):
    if shuffle_list:
        random.shuffle(data)

    n_train = int(len(data) * split)

    train = data[:n_train]
    val = data[n_train:]

    return train, val


def create_annotation_file(data, path_folder, domain, dose_name):
    info = {
        'path_slice': [],
        'partial_path': [],
        'patient': [],
        'domain': [],
        'tube_current': [],
        'bit_stored': []

    }
    for t in data:
        slices_list = sorted(os.listdir(os.path.join(path_folder, t, dose_name)))
        for j in slices_list:
            partial_path = os.path.join(t, dose_name, j)
            total_path = os.path.join(path_folder, partial_path)
            dicom = pydicom.read_file(total_path)
            info['path_slice'].append(total_path)
            info['partial_path'].append(partial_path)
            info['patient'].append(t)
            info['domain'].append(domain)
            try:
                info['tube_current'].append(dicom['XRayTubeCurrent'].value)
            except KeyError:
                info['tube_current'].append('-')
            try:
                info['bit_stored'].append(dicom['BitsStored'].value)
            except KeyError:
                info['bit_stored'].append('-')
    return  pd.DataFrame(info)

if __name__ == "__main__":

    interim_dir = './data/interim/'
    raw_dir = './data/raw/'
    dataset_name = 'mayo-clinic'
    challenge_split = 'Training_Image_Data'
    rec_kernel = '1mm B30'
    modalities = ['full_1mm', 'quarter_1mm']
    patients_path = os.path.join(raw_dir, dataset_name, challenge_split, rec_kernel)

    patients_list = sorted(os.listdir(os.path.join(patients_path, modalities[0])))
    if '.DS_Store' in patients_list:
        patients_list.remove('.DS_Store')
    train_split = 0.7
    val_split = 0.2
    test_split = 0.1
    split_names = ['train', 'val', 'test']

    patients_train, patients_val_test = make_split(patients_list, train_split)
    patients_val, patients_test = make_split(patients_val_test, val_split / (val_split + test_split))

    for split_name, split_data in zip(split_names, [patients_train, patients_val, patients_test]):

        df_ld = create_annotation_file(
            data = split_data,
            path_folder=os.path.join(patients_path, modalities[0]),
            domain='HD',
            dose_name=modalities[0]
        )


        df_hd = create_annotation_file(
            data = split_data,
            path_folder=os.path.join(patients_path, modalities[1]),
            domain='LD',
            dose_name=modalities[1]
        )

        df = pd.concat([df_ld, df_hd], axis=0, ignore_index=True)
        df.to_csv(os.path.join(interim_dir, dataset_name, f'{split_name}-mayo-clinic.csv'))

print('May be the force with you.')

