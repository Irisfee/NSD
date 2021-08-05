#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nilearn import image, masking
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter


def parse_roi_spec(
    roi_id: str, roi_def_dict: dict
) -> tuple[str, Optional[dict], Optional[dict]]:
    """Parse ROI specification for given roi_id."""

    roi_id_split = roi_id.split("-")
    # Check input
    cond1 = (roi_id_split[-1] == "L") or (roi_id_split[-1] == "R")  # ending with L or R
    cond2 = len(roi_id_split) == 1
    cond3 = len(roi_id_split) > 2
    if (not (cond1 or cond2)) or cond3:
        raise ValueError(
            "Argument 'roi_id' should be a string with one and at most one '-'.\n"
            "Before the dash is the ROI name and after dash is the hemisphere indicator (L or R).\n"
            "If this is a bilateral or midline ROI, the hemisphere indicator can be omitted."
        )
    if roi_id_split[0] not in roi_def_dict.keys():
        raise ValueError(
            f"Current roi {roi_id} doesn't have a specification in given roi_spec dict."
        )
    # Generate ROI infomation
    roi_def = roi_def_dict[roi_id_split[0]]
    spec = dict()
    spec["roi_id"] = roi_id
    spec["atlas"] = roi_def["atlas"]
    if roi_id_split[-1] == "L":
        spec["hemi"] = "lh"
        spec["index"] = roi_def["index_lh"]
    elif roi_id_split[-1] == "R":
        spec["hemi"] = "rh"
        spec["index"] = roi_def["index_rh"]
    elif "index_midline" in roi_def.keys():
        spec["hemi"] = "midline"
        spec["index"] = roi_def["index_midline"]
    else:
        spec["hemi"] = "lr"
        spec["index"] = list(set(roi_def["index_lh"] + roi_def["index_rh"]))
        spec["index_lh"] = roi_def["index_lh"]
        spec["index_rh"] = roi_def["index_rh"]
    spec["separate_lr_file"] = roi_def["separate_lr_file"]

    return spec


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


# def gen_mask_from_index(
#     img_list: list[Union[nib.nifti1.Nifti1Image, str, os.PathLike]],
#     index_list: list[list[int]],
# ) -> nib.nifti1.Nifti1Image:
#     """Generate a mask image object based on given indexes."""

#     img_list = [_check_niimg(img) for img in img_list]
#     # Check input image shape and affine
#     for img in img_list:
#         if img.shape != img_list[0].shape:
#             raise Exception("Each input image should be in the same space with same dimensions.")
#         if not np.allclose(img.affine, img_list[0].affine):
#             raise Exception("Each input image should have same affine matrix.")
#     # Get data array for each input image (assume int type)
#     data_list = [img.get_fdata().astype("int16") for img in img_list]
#     mask_data = np.zeros_like(data_list[0])
#     # For each pair of image and index list, find voxels match the index
#     # Then combine all selected voxels togather in one mask image
#     for data, index in zip(data_list, index_list):
#         for idx in index:
#             mask_data += np.where(data == idx, 1, 0)
#     mask_data = np.where(mask_data > 0, 1, 0)
#     mask_img = nib.Nifti1Image(mask_data, img_list[0].affine, img_list[0].header)

#     return mask_img


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


def apply_mask(
    data: Union[
        nib.nifti1.Nifti1Image, h5py._hl.dataset.Dataset, Union[str, os.PathLike]
    ],
    mask_img: nib.nifti1.Nifti1Image,
    data_format: str = "nifti",
) -> np.ndarray:

    if data_format == "nifti":
        return masking.apply_mask(data, mask_img).astype("float32") / 300
    elif data_format == "hdf5":
        if isinstance(data, h5py._hl.dataset.Dataset):
            return _index_hdf5(data, mask_img).astype("float32") / 300
        else:
            with h5py.File(data, mode="r") as ds:
                return _index_hdf5(ds["betas"], mask_img).astype("float32") / 300.0


def _check_niimg(img: Union[nib.nifti1.Nifti1Image, str, os.PathLike]) -> Any:
    """Return nibabel image object."""

    if isinstance(img, (str, os.PathLike)):
        img = nib.load(img)
    return img


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


def pandas2rpy(df: pd.DataFrame) -> ro.vectors.DataFrame:
    """Convert pandas' DataFrame to rpy2's DataFrame."""

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df = ro.conversion.py2rpy(df)

    return df


def rpy2pandas(df: ro.vectors.DataFrame) -> pd.DataFrame:
    """Convert rpy2' DataFrame to pandas's DataFrame."""

    with localconverter(ro.default_converter + ro.pandas2ri.converter):
        df = ro.conversion.rpy2py(df)

    return df