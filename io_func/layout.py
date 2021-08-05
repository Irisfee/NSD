#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Access to NSD dataset."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Union

from .utils import conform_sub_id, check_file_list, _check_space, _check_hemi
from .mask import (
    gen_mask_from_index,
    gen_surf_mask_from_index,
    _index_hdf5,
    _check_niimg,
)


class Layout:
    """NSD project file layout."""

    def __init__(self):
        # Basic directories
#         self.base_dir = Path(project_root_dir)
        self.code_dir = Path("/projects/hulacon/shared/nsd_results/yufei")
        self.data_dir = Path("/projects/hulacon/shared/nsd")
        self.nsddata_dir = self.data_dir.joinpath("nsddata")
        self.nsddata_betas_dir = self.data_dir.joinpath("nsddata_betas", "ppdata")
        self.ppdata_dir = self.nsddata_dir.joinpath("ppdata")
        self.ppdata_sub_dir_prefix = self.nsddata_dir.joinpath("ppdata", "subj{}").as_posix()
        # FreeSurfer directories
        self.fs_dir = self.nsddata_dir.joinpath("freesurfer")
        self.fs_sub_dir_prefix = self.fs_dir.joinpath("subj{}").as_posix()
        # Behavior directories
        self.behav_dir_prefix = self.nsddata_dir.joinpath("ppdata", "subj{}", "behav").as_posix()
        # Stimuli directories
        self.stim_dir = self.nsddata_dir.joinpath("stimuli")
        # Custom directories
        self.deriv_dir = self.code_dir.joinpath("codes_yufei")
        self.roi_dir = self.code_dir.joinpath("roi")
        # Subject list
        self.subject_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
        # Session list
        self.ses_list = {
            "01": list(range(1, 41)),
            "02": list(range(1, 41)),
            "03": list(range(1, 33)),
            "04": list(range(1, 31)),
            "05": list(range(1, 41)),
            "06": list(range(1, 33)),
            "07": list(range(1, 41)),
            "08": list(range(1, 31)),
        }

    """
    Get NSD file (list of filename)
    """

    def get_beh_file(self, sub_id: str) -> list[os.PathLike]:
        """Get behavior responses file."""

        sub_id = conform_sub_id(sub_id)
        file_lst = [self.ppdata_dir.joinpath(f"subj{sub_id}", "behav", "responses.tsv")]
        check_file_list(file_lst, 1)

        return file_lst

    def get_anat_file(
        self, sub_id: str, modality: str = "T1", resolution: str = "0pt8"
    ) -> list[os.PathLike]:
        """Get T1/T2 anatomic file."""

        sub_id = conform_sub_id(sub_id)
        file_lst = [
            self.ppdata_dir.joinpath(
                f"subj{sub_id}", "anat", f"{modality}_{resolution}_masked.nii.gz"
            )
        ]
        check_file_list(file_lst, 1)

        return file_lst

    def get_anat_resampled_file(
        self, sub_id: str, modality: str = "T1", space: str = "func1pt8mm"
    ) -> list[os.PathLike]:
        """Get resampled T1/T2 anatomic file."""

        _check_vol_space(space)
        sub_id = conform_sub_id(sub_id)
        file_lst = [
            self.ppdata_dir.joinpath(f"subj{sub_id}", space, f"{modality}_to_{space}.nii.gz")
        ]
        check_file_list(file_lst, 1)

        return file_lst

    def get_atlas_file(
        self,
        sub_id: str,
        atlas_id: str,
        hemi: str = "LR",
        space: str = "func1pt8mm",
    ) -> list[os.PathLike]:
        """Get atlas file."""

        # Parse input
        sub_id = conform_sub_id(sub_id)
        _check_hemi(hemi, valid_list=["L", "R", "LR"])
        if hemi == "L":
            prefix = "lh."
        elif hemi == "R":
            prefix = "rh."
        else:
            prefix = ""
        # Find file
        if space in ["func1pt8mm", "func1mm"]:
            file_lst = sorted(
                self.ppdata_dir.joinpath(f"subj{sub_id}", space).glob(
                    f"**/{prefix}{atlas_id}.nii.gz"
                )
            )  # some atlas files are not in roi directory, e.g., aseg
        elif space in ["fsaverage", "fsnative"]:
            assert hemi != "LR", "For surface space, hemi must be L or R."
            fs_subj = "fsaverage" if space == "fsaverage" else f"subj{sub_id}"
            file_lst = sorted(
                self.fs_dir.joinpath(fs_subj, "label").glob(f"{prefix}{atlas_id}.mgz")
            )
        else:
            raise ValueError(
                "Unsupported space. Valid choice: func1pt8mm, func1mm, fsaverage, fsnative."
            )
        # Looking for additional atlas
        if (len(file_lst) == 0) and self.roi_dir.joinpath("atlas_list.yaml").is_file():
            raise NotImplementedError
        check_file_list(file_lst, 1)

        return file_lst

    def get_surf_file(
        self,
        sub_id: str,
        hemi: str,
        surf_id: str,
        space: str = "fsnative",
    ) -> list[os.PathLike]:
        """Get surface file inside /surf directory."""

        # Parse input
        sub_id = conform_sub_id(sub_id)
        _check_space(space, valid_list=["fsaverage", "fsnative"])
        _check_hemi(hemi)
        if hemi == "L":
            prefix = "lh"
        elif hemi == "R":
            prefix = "rh"
        # Find file
        fs_subj = "fsaverage" if space == "fsaverage" else f"subj{sub_id}"
        file_lst = sorted(self.fs_dir.joinpath(fs_subj, "surf").glob(f"{prefix}.{surf_id}"))
        check_file_list(file_lst, 1)

        return file_lst

    def get_singletrial_beta_file(
        self,
        sub_id: str,
        ses_list: list[Union[int, str]] = None,
        space: str = "func1pt8mm",
        version: str = "betas_fithrf",
        file_type: str = "nifti",
    ) -> list[os.PathLike]:

        # Parse input
        sub_id = conform_sub_id(sub_id)
        assert ses_list is None or isinstance(ses_list, list)
        _check_space(space)
        assert file_type in ["nifti", "hdf5"], "Valid file_type option: ['nifti', 'hdf5']."
        file_ext = "nii.gz" if file_type == "nifti" else "hdf5"
        if (space == "func1mm") and (file_type == "nifti"):
            raise ValueError("Betas for func1mm space only have hdf5 file.")
        # Find file
        beta_dir = self.nsddata_betas_dir.joinpath(f"subj{sub_id}", space, version)
        if ses_list is None:
            file_lst = sorted(beta_dir.glob(f"betas_session*.{file_ext}"))
            check_file_list(file_lst)
        else:
            file_lst = []
            for ses_id in ses_list:
                file_lst += beta_dir.glob(f"betas_session{int(ses_id):02d}.{file_ext}")
            check_file_list(file_lst, n=len(ses_list))

        return file_lst

    def get_singletrial_beta_surf_file(
        self,
        sub_id: str,
        hemi: str,
        ses_list: list[Union[int, str]] = None,
        space: str = "fsaverage",
        version: str = "betas_fithrf",
    ) -> list[os.PathLike]:

        # Parse input
        sub_id = conform_sub_id(sub_id)
        assert ses_list is None or isinstance(ses_list, list)
        _check_space(space, valid_list=["fsaverage"])
        _check_hemi(hemi)
        if hemi == "L":
            prefix = "lh"
        elif hemi == "R":
            prefix = "rh"
        # Find file
        beta_dir = self.nsddata_betas_dir.joinpath(f"subj{sub_id}", space, version)
        if ses_list is None:
            file_lst = sorted(beta_dir.glob(f"{prefix}.betas_session*.mgh"))
            check_file_list(file_lst)
        else:
            file_lst = []
            for ses_id in ses_list:
                file_lst += beta_dir.glob(f"{prefix}.betas_session{int(ses_id):02d}.mgh")
            check_file_list(file_lst, n=len(ses_list))

        return file_lst


def conform_sub_id(sub_id: str, with_prefix=False) -> str:
    """Conform sub_id."""

    if not isinstance(sub_id, str):
        raise ValueError("Argument sub_id should be a string. (e.g., '01')")
    if (sub_id.startswith("sub-")) and (not with_prefix):
        sub_id = sub_id[4:]
    if (not sub_id.startswith("sub-")) and with_prefix:
        sub_id = f"sub-{sub_id}"

    return sub_id


# def _check_space(space: str, valid_list: list[str] = ["func1pt8mm", "func1mm"]):
#     """Check input argument space."""
#     assert space in valid_list, f"Valid space option: {valid_list}."


def _check_vol_space(space: str, valid_list: list[str] = ["func1pt8mm", "func1mm"]):
    """Check input argument space."""
    assert space in valid_list, f"Valid space option: {valid_list}."


def _check_surf_space(space: str, valid_list: list[str] = ["fsaverage", "fsnative"]):
    """Check argument space."""
    assert space in valid_list, f"Valid space option: {valid_list}."

