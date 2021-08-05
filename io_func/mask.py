#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relates to ROI."""


from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nilearn import image, masking


def gen_mask_from_index(
    img: Union[nib.nifti1.Nifti1Image, str, os.PathLike],
    index_list: list[int],
) -> nib.nifti1.Nifti1Image:
    """Generate a mask image object based on given indexes."""

    # Check input
    img = _check_niimg(img)
    # Get data array from input atlas image
    img_data = img.get_fdata().astype("int16")
    # Find voxels based on given index list and make a single image object
    mask_data = np.zeros_like(img_data)
    for idx in index_list:
        mask_data += np.where(img_data == idx, 1, 0)
    mask_data = np.where(mask_data > 0, 1, 0)
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)

    return mask_img


def gen_surf_mask_from_index(
    img: Union[nib.freesurfer.mghformat.MGHImage, str, os.PathLike],
    index_list: list[int],
) -> nib.freesurfer.mghformat.MGHImage:
    """Generate a surface mask image from atlas image based on given indexes."""

    # Check input
    img = _check_niimg(img)
    # Get data array from input atlas image
    img_data = img.get_fdata().astype("int16")
    # Find vertices based on given index list and make a single image object
    mask_data = np.zeros_like(img_data)
    for idx in index_list:
        mask_data += np.where(img_data == idx, 1, 0)
    mask_data = np.where(mask_data > 0, 1, 0)
    mask_img = nib.freesurfer.mghformat.MGHImage(mask_data, img.affine, img.header)

    return mask_img


def _index_hdf5(dataset, mask_img):
    """Index 3d roi data from hdf5 dataset into 2-D array."""

    data = np.zeros((dataset.shape[0], np.count_nonzero(mask_img.get_fdata())))
    coords = mask_img.get_fdata().astype("bool").nonzero()
    # Note: NSD's hdf5 file saved from MATLAB which results a reversed dimension
    # order in python. i.e., the first dimension in beta is the 4th dimension in the
    # NIFIT file
    for idx, (i, j, k) in enumerate(zip(coords[0], coords[1], coords[2])):
        data[:, idx] = dataset[:, k, j, i]

    return data


def _check_niimg(img: Union[nib.nifti1.Nifti1Image, str, os.PathLike]) -> Any:
    """Return nibabel image object."""

    if isinstance(img, (str, os.PathLike)):
        img = nib.load(img)
    return img