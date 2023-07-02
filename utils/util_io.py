import pickle
import datetime
import zipfile
import glob
import os

from . import util_path

def write_excel(*keys, dictionary, sheet_idx, col_position=1, wb=None, ws=None, default='not found'):
    """ Function to save the patients data splits (ids and labels) to file excel (openpyxl library is used)
    Args:
        keys: string values representing the field from the dictionary to save in excel
        dictionary: dictionary representing the data
        sheet_idx: int value representing the index of the current worksheet
        col_position: int value representing the coordinate of the column to begin to start the writing
        wb: existing workbook
        ws: existing worksheet
        default: default values to return if the inserted key do not exist in dictionary
    Returns:
        wb: workbook
        ws: worksheet
    """

    if wb is None:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '_fold' + str(sheet_idx)
    if ws is None:
        ws = wb.create_sheet(title='_fold' + str(sheet_idx))

    ws.cell(row=1, column=col_position, value=keys[0].split('_')[1] + '_id')
    ws.cell(row=1, column=col_position + 1, value=keys[0].split('_')[1] + '_label')

    for idx in range(dictionary.get(keys[0], default).shape[0]):
        [ws.cell(row=idx + 2, column=col_position + column, value=dictionary.get(key, default)[idx, 0]) for key, column in zip(keys, np.arange(len(keys)))]
    return wb, ws

def get_string_datetime():
    now = datetime.datetime.now()
    if now.month < 10:
        month_string = '0'+str(now.month)
    else:
        month_string = str(now.month)
    if now.day < 10:
        day_string = '0'+str(now.day)
    else:
        day_string = str(now.day)
    yearmonthdate_string = str(now.year) + month_string + day_string
    return yearmonthdate_string


def write_list_to_file(my_list, path):
    with open(path, 'w+') as f:
        for item in my_list:
            f.write("%s\n" % item)


def read_file_to_list(path):
    with open(path, 'r') as f:
        x = f.readlines()
    return x


def write_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def add_to_zip(zipObj, patient, split):
    files = glob.glob(os.path.join(patient, "*.pickle"))
    if len(files) == 0:
        files = glob.glob(os.path.join(patient, "*.png"))

    print(f">> Writing {patient} to zip file")
    for file in files:
        filename = os.path.join(
            split,
            util_path.get_filename_without_extension(patient),
            util_path.get_filename(file),
        )
        # Add file to zip
        zipObj.write(file, filename)