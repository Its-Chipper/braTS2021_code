import pickle
import os
import numpy as np
import nibabel as nib

base_folder_root = "D:/Research/Spring_25/archive/Data/"

file_extensions = ('.nii.gz')

file_suffixes = ('flair', 'seg', 't1', 't1ce', 't2')

def main_folder_loop():
    for (root,dirs,files) in os.walk(base_folder_root, topdown=True):
        for file in files:
            file_data = file_reader(root + '/' + file)
            normalisation(file_data)
            break

def file_reader(file_root):
    if not os.path.exists(file_root):
        print("Couldn't find file (%s)" % (file_root))

    proxy = nib.load(file_root)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def file_saver(file_path, file_data):
    with open (file_path, 'wb') as file:
        pickle.dump(file_data, file)

def normalisation(data):
    return data/np.linalg.norm(data)

if __name__ == "__main__":
    main_folder_loop()