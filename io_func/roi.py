#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import annotations
from typing import Optional
import os
import numpy as np
import h5py
import nibabel as nib


def parse_roi_id(roi_id: str) -> tuple[str, str]:
    """Parse ROI ID."""

    parts = roi_id.split("-")
    cond1 = (parts[-1] == "L") or (parts[-1] == "R")  # ending with L or R
    cond2 = len(parts) == 1
    cond3 = len(parts) > 2
    # Ensure roi_id is ending with 'L', 'R' with exactly one '-'
    # Or roi_id doesn't contain any '-' (means bilateral or midline ROI)
    if (not (cond1 or cond2)) or cond3:
        raise ValueError(
            "Argument 'roi_id' should be a string with one and at most one '-'.\n"
            "Before the dash is the ROI name and after dash is the hemisphere indicator (L or R).\n"
            "If this is a bilateral or midline ROI, the hemisphere indicator should be omitted."
        )
    # ROI information
    roi_name = parts[0]
    hemi = parts[1] if len(parts) == 2 else "LR"

    return (roi_name, hemi)


def make_mask_from_index(data: np.ndarray, index_list: list[int]) -> np.ndarray:
    """Make a binary mask array from an index array."""

    mask = np.zeros_like(data)
    for idx in index_list:
        mask += np.where(data == idx, 1, 0)
    mask = np.where(mask > 0, 1, 0)
    return mask


def make_roi_from_spec(
    roi_id: str, roi_spec: dict, atlas_file: os.PathLike
) -> nib.nifti1.Nifti1Image:
    """Generate ROI (volume) mask from an atlas file."""

    # Parse ROI information
    roi_id, hemi = parse_roi_id(roi_id)
    spec = roi_spec[roi_id]
    # Make mask image
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(np.int16)
    if hemi == "L":
        mask_data = make_mask_from_index(atlas_data, spec["index_L"])
    if hemi == "R":
        mask_data = make_mask_from_index(atlas_data, spec["index_R"])
    if hemi == "LR":
        mask_lh = make_mask_from_index(atlas_data, spec["index_L"])
        mask_rh = make_mask_from_index(atlas_data, spec["index_R"])
        mask_data = mask_lh + mask_rh
    mask_data = np.where(mask_data > 0, 1, 0)
    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)

    return mask_img


def make_roi_surf_from_spec(
    roi_id: str,
    roi_spec: dict,
    atlas_file_list: list[Optional[os.PathLike], Optional[os.PathLike]],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate ROI (surface) mask from an atlas file."""

    # Parse input
    assert (
        len(atlas_file_list) == 2
    ), "atlas_file_list should be a list with atlas files for left and right hemisphere."
    roi_id, hemi = parse_roi_id(roi_id)
    spec = roi_spec[roi_id]
    # Make mask data array
    if hemi == "L":
        atlas_data = np.squeeze(nib.load(atlas_file_list[0]).get_fdata().astype(np.int16))
        mask_data = make_mask_from_index(atlas_data, spec["index_L"])
        out_mask = (mask_data.astype(np.bool), None)
    if hemi == "R":
        atlas_data = np.squeeze(nib.load(atlas_file_list[1]).get_fdata().astype(np.int16))
        mask_data = make_mask_from_index(atlas_data, spec["index_R"])
        out_mask = (None, mask_data.astype(np.bool))
    if hemi == "LR":
        atlas_data_lh = np.squeeze(nib.load(atlas_file_list[0]).get_fdata().astype(np.int16))
        mask_data_lh = make_mask_from_index(atlas_data_lh, spec["index_L"])
        atlas_data_rh = np.squeeze(nib.load(atlas_file_list[1]).get_fdata().astype(np.int16))
        mask_data_rh = make_mask_from_index(atlas_data_rh, spec["index_R"])
        out_mask = (mask_data_lh.astype(np.bool), mask_data_rh.astype(np.bool))

    return out_mask


def loop_index_hdf5(
    dataset: h5py._hl.dataset.Dataset, mask_img: nib.nifti1.Nifti1Image
) -> np.ndarray:
    """Index 3d ROI data from hdf5 dataset into 2-D array."""

    data = np.zeros((dataset.shape[0], np.count_nonzero(mask_img.get_fdata())))
    coords = mask_img.get_fdata().astype("bool").nonzero()
    # Note: NSD's hdf5 file saved from MATLAB which results a reversed dimension
    # order in python. i.e., the first dimension in beta is the 4th dimension in the
    # NIFIT file
    for idx, (i, j, k) in enumerate(zip(coords[0], coords[1], coords[2])):
        data[:, idx] = dataset[:, k, j, i]

    return data
