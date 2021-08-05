#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Project data io class."""


from __future__ import annotations
import os
from typing import Optional, Union
import warnings
import yaml
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import nilearn.image as nli
import nilearn.masking as nlm

from .layout import Layout
from .roi import parse_roi_id, make_roi_from_spec, make_roi_surf_from_spec, loop_index_hdf5
from .utils import _check_space, _check_hemi, conform_sub_id


class Project(Layout):
    """Common project data manipulation."""

    def __init__(self):
        super().__init__()

    ###################
    # Read common files
    ###################

    def read_beh_raw(self, sub_id: str) -> pd.DataFrame:
        """Read raw behavior responses file."""

        df = pd.read_csv(
            self.get_beh_file(sub_id)[0],
            sep="\t",
            dtype={
                "SUBJECT": "Int64",
                "SESSION": "Int64",
                "RUN": "Int64",
                "TRIAL": "Int64",
                "73KID": "Int64",
                "10KID": "Int64",
                "TIME": "float",
                "ISOLD": "Int64",
                "ISCORRECT": "Int64",
                "RT": "float",
                "CHANGEMIND": "Int64",
                "MEMORYRECENT": "Int64",
                "MEMORYFIRST": "Int64",
                "ISOLDCURRENT": "Int64",
                "ISCORRECTCURRENT": "Int64",
                "TOTAL1": "Int64",
                "TOTAL2": "Int64",
                "BUTTON": "Int64",
                "MISSINGDATA": "Int64",
            },
        )

        return df

    def read_beh(self, sub_id: str) -> pd.DataFrame:
        """Read behavior responses file."""

        # Read raw behavior result
        df = self.read_beh_raw(sub_id)
        # Rename columns
        df = df.rename(
            columns={
                "SUBJECT": "sub_id",
                "SESSION": "ses_id",
                "RUN": "run_id",
                "TRIAL": "trial_id",
                "73KID": "stim_73k_id",
                "10KID": "stim_10k_id",
                "TIME": "offset_time",
                "ISOLD": "is_old",
                "ISCORRECT": "is_correct",
                "RT": "resp_time",
                "CHANGEMIND": "change_mind",
                "MEMORYRECENT": "lag_to_recent",
                "MEMORYFIRST": "lag_to_first",
                "ISOLDCURRENT": "is_old_in_session",
                "ISCORRECTCURRENT": "is_correct_in_session",
                "TOTAL1": "num_resp_new",
                "TOTAL2": "num_resp_old",
                "BUTTON": "button_pressed",
                "MISSINGDATA": "missing_data",
            }
        )
        # Padding subject id
        df["sub_id"] = df["sub_id"].astype("string").str.pad(width=2, side="left", fillchar="0")
        # Convert offset_time from MATLAB's serial data number to pandas' timedelta
        # The onset time of the first trial in the first session was set to 0
        df["offset_time"] = (
            pd.to_datetime(df["offset_time"], unit="D").diff().fillna(pd.to_timedelta(0)).cumsum()
        )
        # Calculate repetition information
        df["rep_id"] = df.groupby(["sub_id", "stim_73k_id"]).cumcount() + 1
        df["rep_total"] = df.groupby(["sub_id", "stim_73k_id"])["stim_73k_id"].transform("count")
        # Get shared stimuli infomation
        df_shared = pd.read_csv(
            self.nsddata_dir.joinpath("stimuli", "nsd", "shared1000.tsv"),
            sep="\t",
            header=None,
            names=["stim_73k_id"],
        )
        df_shared["is_shared"] = 1
        df_shared["is_shared"] = df_shared["is_shared"].astype("Int64")
        df = pd.merge(df, df_shared, how="left", on="stim_73k_id").fillna({"is_shared": 0})
        # Calculate response type
        df["resp_type"] = pd.NA
        df.loc[(df["is_old"] == 1) & (df["is_correct"] == 1), "resp_type"] = "hit"
        df.loc[(df["is_old"] == 1) & (df["is_correct"] == 0), "resp_type"] = "miss"
        df.loc[(df["is_old"] == 0) & (df["is_correct"] == 1), "resp_type"] = "cr"
        df.loc[(df["is_old"] == 0) & (df["is_correct"] == 0), "resp_type"] = "fa"
        # Calculate response
        df["is_old_resp"] = pd.NA
        df.loc[df["resp_type"].isin(["hit", "fa"]), "is_old_resp"] = 1
        df.loc[df["resp_type"].isin(["miss", "cr"]), "is_old_resp"] = 0
        # Reorder columns
        df = df[
            [
                "sub_id",
                "ses_id",
                "run_id",
                "trial_id",
                "stim_73k_id",
                "stim_10k_id",
                "is_shared",
                "offset_time",
                "is_old",
                "is_old_resp",
                "is_correct",
                "resp_type",
                "resp_time",
                "rep_id",
                "rep_total",
                "lag_to_first",
                "lag_to_recent",
                "num_resp_new",
                "num_resp_old",
                "is_old_in_session",
                "is_correct_in_session",
                "missing_data",
                "change_mind",
                "button_pressed",
            ]
        ]

        return df

    ###################################
    # Read singletrial estimation files
    ###################################

    def read_singletrial_beta(
        self,
        sub_id: str,
        ses_list: Optional[list[Union[int, str]]] = None,
        space: str = "func1pt8mm",
        hemi: str = "LR",
        version: str = "betas_fithrf",
        file_type: str = "nifti",
    ) -> np.ndarray:
        """Read beta data."""

        # Parse input
        _check_space(space, valid_list=["func1pt8mm", "func1mm", "fsaverage"])
        assert file_type in ["nifti", "hdf5"], "Valid file_type option: ['nifti', 'hdf5']."
        if (space == "func1mm") and (file_type == "nifti"):
            raise ValueError("Betas for func1mm space only have hdf5 file.")
        # Surface space
        if space == "fsaverage":
            _check_hemi(hemi)
            file_lst = self.get_singletrial_beta_surf_file(
                sub_id, hemi, ses_list=ses_list, space=space, version=version
            )
            data = []
            for f in file_lst:
                data.append(np.squeeze(nib.load(f).get_fdata(dtype=np.float32)).T)
            data = np.concatenate(data, axis=0)
        # Volume space
        else:
            file_lst = self.get_singletrial_beta_file(
                sub_id, ses_list=ses_list, space=space, version=version, file_type=file_type
            )
            if file_type == "nifti":
                # Workaround, since nilearn doesn't accept pathlib object for now
                file_lst = [fid.as_posix() for fid in file_lst]
                data = nli.load_img(file_lst).get_fdata(dtype=np.float32) / 300
            else:
                data = []
                for f in file_lst:
                    with h5py.File(f, mode="r") as ds:
                        data.append(np.array(ds["betas"]).T.astype("float32") / 300)
                data = np.concatenate(data, axis=3)

        return data

    def read_singletrial_beta_roi(
        self,
        sub_id: str,
        roi_mask: dict,
        ses_list: Optional[list[Union[int, str]]] = None,
        space: str = "func1pt8mm",
        version: str = "betas_fithrf",
        file_type: str = "nifti",
        return_lr_separete: bool = False,
    ) -> Union[list[np.ndarray], list[np.ndarray, np.ndarray]]:

        # Parse input
        _check_space(space, valid_list=["func1pt8mm", "func1mm", "fsaverage", "fsnative"])
        # Volume space
        if space in ["func1pt8mm", "func1mm"]:
            assert roi_mask["Volume"] is not None, f"Volume ROI mask not found for space-{space}."
            # Get beta file list
            file_lst = self.get_singletrial_beta_file(
                sub_id, ses_list=ses_list, space=space, version=version, file_type=file_type
            )
            if file_type == "nifti":
                file_lst = [f.as_posix() for f in file_lst]
                data = nlm.apply_mask(file_lst, roi_mask["Volume"]).astype("float32") / 300
            elif file_type == "hdf5":
                data = []
                for f in file_lst:
                    with h5py.File(f, mode="r") as ds:
                        data_ses = loop_index_hdf5(ds["betas"], roi_mask["Volume"])
                        data_ses = data_ses.astype("float32") / 300
                        data.append(data_ses)
                data = np.concatenate(data, axis=0)
        # Surface space
        if space == "fsaverage":
            assert not (
                (roi_mask["Surface_L"] is None) and (roi_mask["Surface_R"] is None)
            ), f"Surface ROI mask not found for space-{space}."
            # Left hemisphere
            if roi_mask["Surface_L"] is not None:
                # Read beta data
                file_lst = self.get_singletrial_beta_surf_file(
                    sub_id, "L", ses_list=ses_list, space=space, version=version
                )
                data_lh = []
                for f in file_lst:
                    data_ses = np.squeeze(nib.load(f).get_fdata().astype("float32")).T
                    data_ses = data_ses[:, roi_mask["Surface_L"].astype(np.bool)]
                    data_lh.append(data_ses)
                data_lh = np.concatenate(data_lh, axis=0)
            else:
                data_lh = None
            if roi_mask["Surface_R"] is not None:
                # Read beta data
                file_lst = self.get_singletrial_beta_surf_file(
                    sub_id, "R", ses_list=ses_list, space=space, version=version
                )
                data_rh = []
                for f in file_lst:
                    data_ses = np.squeeze(nib.load(f).get_fdata().astype("float32")).T
                    data_ses = data_ses[:, roi_mask["Surface_R"].astype(np.bool)]
                    data_rh.append(data_ses)
                data_rh = np.concatenate(data_rh, axis=0)
            else:
                data_rh = None
            # Combine data
            data = [data_lh, data_rh]
            if not return_lr_separete:
                data = [i for i in data if i is not None]
                data = np.concatenate(data, axis=1)

        return data

    #############
    # ROI related
    #############
    def read_roi_spec(self, roi_spec_file: Optional[os.PathLike] = None) -> dict:
        """Read atlas list file."""

        if roi_spec_file is None:
            # Use default file if not provided
            roi_spec_file = self.roi_dir.joinpath("roi_definition.yaml")
        with open(roi_spec_file, "r") as f:
            roi_spec = yaml.load(f, Loader=yaml.CLoader)

        return roi_spec

    def read_roi_list(self, roi_list_file: Optional[os.PathLike] = None) -> dict:
        """Read atlas list file."""

        if roi_list_file is None:
            # Use default file if not provided
            roi_list_file = self.roi_dir.joinpath("roi_list.yaml")
        with open(roi_list_file, "r") as f:
            roi_list = yaml.load(f, Loader=yaml.CLoader)

        return roi_list

    def make_roi_from_spec(
        self,
        sub_id: str,
        roi_id: str,
        roi_spec: dict,
        space: str = "func1pt8mm",
    ) -> dict:
        """Make ROI mask from an atlas."""

        # Parse input
        _check_space(space, valid_list=["func1pt8mm", "func1mm", "fsaverage", "fsnative"])
        roi_name, _ = parse_roi_id(roi_id)
        # Make ROI
        roi_mask = {"Volume": None, "Surface_L": None, "Surface_R": None}
        # Volume
        if space in ["func1pt8mm", "func1mm"]:
            atlas_file = self.get_atlas_file(sub_id, roi_spec[roi_name]["atlas"], space=space)[0]
            roi_mask["Volume"] = make_roi_from_spec(roi_id, roi_spec, atlas_file)
        # Surface
        else:
            atlas_file_lst = [
                self.get_atlas_file(sub_id, roi_spec[roi_name]["atlas"], hemi="L", space=space)[0],
                self.get_atlas_file(sub_id, roi_spec[roi_name]["atlas"], hemi="R", space=space)[0],
            ]
            surf_mask = make_roi_surf_from_spec(roi_id, roi_spec, atlas_file_lst)
            roi_mask["Surface_L"] = surf_mask[0]
            roi_mask["Surface_R"] = surf_mask[1]

        return roi_mask

    ##########################
    # Quick plotting functions
    ##########################
    def view_img(
        self,
        stat_map_img: nib.nifti1.Nifti1Image,
        sub_id: Optional[str] = None,
        space: str = "func1pt8mm",
        bg_img: Union[nib.nifti1.Nifti1Image, str] = "T1",
        **kwargs,
    ):
        """View volume image."""

        import nilearn.plotting as nlp

        if sub_id:
            # Get bg_img
            if isinstance(bg_img, str):
                if bg_img in ["T1", "T2"]:
                    bg_img = self.get_anat_file(sub_id, modality=bg_img)[0]
                elif bg_img in ["resampled_T1", "resampled_T2"]:
                    bg_img = self.get_anat_resampled_file(sub_id, modality=bg_img, space=space)
                elif bg_img in ["meanBOLD"]:
                    bg_img = self.ppdata_dir.joinpath(
                        f"subj{conform_sub_id(sub_id)}", space, "mean.nii.gz"
                    )
                else:
                    raise ValueError(
                        "Valid bg_img: ['T1', 'T2', 'resampled_T1', 'resampled_T2', 'meanBOLD']"
                    )
        else:
            # If sub_id is not provided, use MNI152 template as default
            if not isinstance(bg_img, (os.PathLike, nib.nifti1.Nifti1Image)):
                bg_img = "MNI152"
        with warnings.catch_warnings():  # Ignore nilearn's UserWarning
            warnings.simplefilter("ignore")
            g = nlp.view_img(stat_map_img, bg_img=str(bg_img), **kwargs)

        return g

    def view_surf_fs_data(
        self,
        surf_map: Union[np.ndarray, str],
        hemi: str,
        sub_id: Optional[str] = None,
        surf_mesh: Optional[Union[os.PathLike, list[np.ndarray, np.ndarray]]] = None,
        surf_mesh_name: str = "inflated",
        bg_map: Optional[Union[str, np.ndarray]] = "sulc",
        **kwargs,
    ):
        """View surface data."""

        import nilearn.plotting as nlp

        space = "fsnative" if sub_id else "fsaverage"
        fs_sub_id = sub_id if sub_id else "fsaverage"
        if surf_mesh is None:
            surf_mesh = str(self.get_surf_file(fs_sub_id, hemi, surf_mesh_name, space=space)[0])
        elif isinstance(surf_mesh, os.PathLike):
            surf_mesh = str(surf_mesh)
        if isinstance(bg_map, str):
            bg_map = str(self.get_surf_file(fs_sub_id, hemi, bg_map, space=space)[0])
        elif isinstance(bg_map, str):
            bg_map = str(bg_map)
        with warnings.catch_warnings():  # Ignore nilearn's UserWarning
            warnings.simplefilter("ignore")
            g = nlp.view_surf(surf_mesh, surf_map=surf_map, bg_map=bg_map, **kwargs)

        return g

    def view_roi(
        self,
        stat_map_img: nib.nifti1.Nifti1Image,
        sub_id: Optional[str] = None,
        space: str = "func1pt8mm",
        bg_img: Union[nib.nifti1.Nifti1Image, str] = "T1",
        threshold: float = 0.1,
        cmap: str = "Set1",
        symmetric_cmap: bool = False,
        colorbar: bool = False,
        vmax: float = 10,
        **kwargs,
    ):
        """View surface ROI."""

        return self.view_img(
            stat_map_img,
            sub_id=sub_id,
            space=space,
            bg_img=bg_img,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            vmax=vmax,
            **kwargs,
        )

    def view_surf_fs_roi(
        self,
        surf_map: Union[np.ndarray, str],
        hemi: str,
        sub_id: Optional[str] = None,
        surf_mesh: Optional[Union[os.PathLike, list[np.ndarray, np.ndarray]]] = None,
        surf_mesh_name: str = "inflated",
        bg_map: Optional[Union[str, np.ndarray]] = "sulc",
        threshold: float = 0.1,
        cmap: str = "Set1",
        symmetric_cmap: bool = False,
        colorbar: bool = False,
        vmax: float = 10,
        **kwargs,
    ):
        """View surface ROI."""

        return self.view_surf_fs_data(
            surf_map,
            hemi,
            sub_id=sub_id,
            surf_mesh=surf_mesh,
            surf_mesh_name=surf_mesh_name,
            bg_map=bg_map,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            vmax=vmax,
            **kwargs,
        )
