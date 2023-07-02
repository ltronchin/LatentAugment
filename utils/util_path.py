import os
import ntpath
from pathlib import Path
import glob

def create_dir(outdir): # function to create directory
    if not os.path.exists(outdir):
        Path(outdir).mkdir(parents=True, exist_ok=True) # with parents 'True' creates all tree/nested folder

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def listdir_nohidden_with_path(path):
    return glob.glob(os.path.join(path, '*'))

def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders

def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))

def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]