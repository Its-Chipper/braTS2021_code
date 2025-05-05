import pickle
import os
import torch
import torchio
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

base_folder_root = "D:/Research/Spring_25/archive/Data/"

file_extensions = ['.nii.gz']

train_file_suffixes = ['flair', 't1', 't1ce', 't2']
gt_file_suffixes = ['seg']

center_crop_dimensions = [160, 160, 100]

transform = torchio.Compose([
    torchio.RandomFlip(axes=(0,), p=1),
    torchio.RandomFlip(axes=(1,), p=0.5),
    torchio.CropOrPad((160, 160, 100)),
    torchio.Resample(0.75),
    torchio.OneOf({
        torchio.RandomElasticDeformation(): 0.6,
        torchio.RandomAffine(): 0.4
    }),
    torchio.OneOf({
        torchio.RandomBlur(): 0.6,
        torchio.Lambda(lambda x: x): 0.4
    }),
    torchio.OneOf({
        torchio.RandomNoise(mean=128, std=10): 0.5,
        torchio.RescaleIntensity(out_min_max=(0, 1)): 0.5
    }),
    torchio.ZNormalization()
])


def main_folder_loop():
    for (root,dirs,files) in os.walk(base_folder_root, topdown=True):
        for file in files:
            file_data = file_reader(root + '/' + file)
            file_data = torchio.ScalarImage(root + '/' + file)
            processed_data = preprocessing(file_data)

            # show images
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))

            slice_idx = 65

            imgs = [
                (file_data.data.numpy()[0, :, :, slice_idx], "Original FLAIR MRI"),
                (processed_data.data.numpy()[0, :, :, slice_idx], "Transformed Image")
            ]

            for i, (img, title) in enumerate(imgs):
                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(title)
                axes[i].axis("off")  
            plt.show()
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

def center_crop(crop_data):
    mid_points = (crop_data.shape[0]//2, crop_data.shape[1]//2, crop_data.shape[2]//2)
    center_crop_dimensions_half = [val // 2 for val in center_crop_dimensions]
    crop_data = crop_data[
        mid_points[0] - center_crop_dimensions_half[0]:mid_points[0] + center_crop_dimensions_half[0],
        mid_points[1] - center_crop_dimensions_half[1]:mid_points[1] + center_crop_dimensions_half[1],
        mid_points[2] - center_crop_dimensions_half[2]:mid_points[2] + center_crop_dimensions_half[2]
    ]
    return crop_data

def preprocessing(data):
    return transform(data)

if __name__ == "__main__":
    main_folder_loop()