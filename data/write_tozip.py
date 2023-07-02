import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import sys
import zipfile
import click
from typing import Any, Optional, Tuple, Union

import glob

from genlib.utils import util_general

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

def write_to_zip(
        source:                     str,
        dest:                       str,
        basename:                   str,
        phase:                      str,
):

    def add_to_zip(zipObj, patient, split):
        files = glob.glob(os.path.join(patient, "*.pickle"))
        if len(files) == 0:
            files = glob.glob(os.path.join(patient, "*.png"))

        print(f">> Writing {patient} to zip file")
        for file in files:
            filename = os.path.join(
                split,
                util_general.get_filename_without_extension(patient),
                util_general.get_filename(file),
            )
            # Add file to zip
            zipObj.write(file, filename)

    # Add basename.
    source_path = os.path.join(source, basename)

    # Get all patients in temp folder.
    patients = glob.glob(os.path.join(source_path, "*[!json]"))
    # Get only the names of patients.
    patients = [util_general.get_filename_without_extension(patient) for patient in patients]
    assert len(patients) > 0

    # Init zip file.
    out_path = os.path.join(dest, f"{basename}.zip",)

    # Write to zip
    with zipfile.ZipFile(out_path, "w") as zipObj:
        for p in patients:
            patient_path = os.path.join(source_path, p)
            add_to_zip(zipObj, patient_path, phase)

@click.command()
@click.option('--interim_dir', help='Output directory for output dataset', required=True, metavar='PATH')
@click.option('--dataset',  help='Name of the input dataset', required=True, type=str)
@click.option('--phase',    help='train/test/val split to save', required=True, type=str)
@click.option('--basename', help='Filename of zipfile containing the inverted latent code', required=True, type=str)
def main(**kwargs):
    opts = EasyDict(**kwargs)
    basename=opts.basename
    phase=opts.phase

    print('Create file and directory')
    source_dir = os.path.join(opts.interim_dir, opts.dataset, 'temp-projector')
    dest_dir = os.path.join(opts.interim_dir, opts.dataset)

    # Useful print
    print()
    print(f'Data directory:      {source_dir}')
    print(f'Output directory:    {dest_dir}')
    print(f'Basename:            {basename}')

    print(f"\nSave to zip, output folder: {dest_dir}")
    write_to_zip(source=source_dir, dest=dest_dir, basename=basename, phase=phase)

if __name__ == "__main__":
    main()