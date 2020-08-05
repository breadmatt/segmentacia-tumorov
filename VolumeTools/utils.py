import numpy as np
from glob import glob
import os
import fnmatch
import SimpleITK as sitk
import scipy.misc as msc


def normalize0_1(array):
    flatten = array.flatten()
    min = np.min(flatten)
    max = np.max(flatten)
    normalized = flatten - min
    normalized/=(max - min)

    return normalized.reshape([155, 240, 240])

def normalize0_255(array):
    array = array * 255.0/np.max(array)
    
    return array.astype(np.uint8)

def binarize_labels(image):
    return sitk.GetArrayFromImage(sitk.Cast(sitk.Equal(image, 0, 1, 0), sitk.sitkFloat32))

def normalize_volumes(volumes):
    return list(map(lambda x: normalize0_1(sitk.GetArrayFromImage(x)), volumes))

def binarize_labels_list(labels):
    return list(map(lambda x: sitk.GetArrayFromImage(sitk.Cast(sitk.Equal(x, 0, 1, 0), sitk.sitkFloat32)), labels))


def get_volume_paths(input_data_dir, vol_types):

    samples = glob(input_data_dir + "*/")
    volume_paths = []

    for dir in samples:
        for dname in os.listdir(dir):
            path = os.path.join(dir, dname)

            if os.path.isdir(path) and any(fnmatch.fnmatch(path, '*'+t+'*') for t in vol_types): #  fnmatch.fnmatch(path, '*'+vol_type+'*'):
                for fname in os.listdir(path):
                    volume_path  = os.path.join(path, fname)
                    if fnmatch.fnmatch(volume_path, '*.mha'):
                        volume_paths.append(volume_path)

    return volume_paths

def load_volume(volume_path, isLabels = False):
    if isLabels:
        castTo = sitk.sitkUInt8
    else:
        castTo = sitk.sitkFloat32
    return sitk.ReadImage(volume_path, castTo)

def load_volumes(volume_paths, isLabels = False):
    if isLabels:
        castTo = sitk.sitkUInt8
    else:
        castTo = sitk.sitkFloat32
    return list(map(lambda x: sitk.ReadImage(x, castTo), volume_paths))

def save_slices_as_bmp(array, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(0, array.shape[0]):
        a = array[i]
        msc.imsave(path + f"{i}.jpg", a)