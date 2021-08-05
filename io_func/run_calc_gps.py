#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np

from jomungard.io.project import Project
from jomungard.global_matching.feature import calc_global_pattern_similarity
from bifrost import slurm
from bifrost.parser import add_slurm_argument
from bifrost.utils import convert_to_gb


# Get parameters
# yapf: disable
parser = argparse.ArgumentParser(description="Parameters.")
parser.add_argument(
    "--roi_id",
    action="store",
    nargs="*",
    help="ROI identifier."
)
parser = add_slurm_argument(parser)
args = parser.parse_args()
# yapf: enable

# Slurm parameters
# Parallelization
nprocs = max(args.cpus_per_task, 4)
# Memory limit
mem = f"{max(convert_to_gb(args.mem), 32)}GB"

# Project
proj = Project(os.getenv("PROJECT_ROOT"))

# Read ROI definition
roi_spec = proj.read_roi_spec()
# Read ROI list
roi_info = proj.read_roi_list()

# Parameters
# Subject list
subject_list = proj.subject_list
# ROI list
if args.roi_id is not None:
    roi_list = args.roi_id
else:
    roi_list = roi_info.keys()

# Submit or run the script
if args.submit:
    jobs = slurm.Slurm(
        account=args.account,
        partition=args.partition,
        job_name="CalcGPS",
        time=args.time,
        n_cpus=nprocs,
        n_mem=mem,
        chdir=proj.base_dir,
    )
    cmd_lst = [
        f"singularity exec -e ${{CONTAINER_DIR}}/valhalla.sif "
        f"python {__file__} --roi_id ${{ROI_ID}} ",
    ]
    jobs.submit_command(cmd_lst, array_list=roi_list, array_index="ROI_ID")
    print(jobs.job_script[0])
else:
    # Loop for subject
    for sub_id in subject_list:
        feature_dir = proj.deriv_dir.joinpath("global_matching", "beh_3bin", f"sub-{sub_id}")
        out_dir = proj.deriv_dir.joinpath("global_matching", "gps_3bin", f"sub-{sub_id}")
        out_dir.mkdir(exist_ok=True, parents=True)
        # Read behavior information
        with open(feature_dir.joinpath(f"sub-{sub_id}_beh_info.pkl"), "rb") as f:
            param = pickle.load(f)
        # Read trial mask
        with open(feature_dir.joinpath(f"sub-{sub_id}_beh_mask.pkl"), "rb") as f:
            mask = pickle.load(f)
        # Loop for ROI
        for roi_id in roi_list:
            print(f"Calculating sub-{sub_id} roi-{roi_id}...", flush=True)
            # ROI space
            space = roi_info[roi_id]["space"]
            # Make ROI mask
            roi_mask = proj.make_roi_from_spec(sub_id, roi_id, roi_spec, space=space)
            # Read beta data
            beta_data = proj.read_singletrial_beta_roi(
                sub_id,
                roi_mask,
                ses_list=list(
                    range(
                        (param["ses_min"] - param["ses_ext"]),
                        (param["ses_max"] + param["ses_ext"]) + 1,
                    )
                ),
                space=space,
                file_type="hdf5",
            )
            beta_data = beta_data[param["beh"].index, :]
            # Remove pontential Nan value (any column/voxel contains Nan)
            beta_data = beta_data[:, ~np.isnan(beta_data).any(axis=0)]
            # Calculate GPS
            gps = calc_global_pattern_similarity(beta_data, param, mask, nprocs)
            # Save result to disk
            out_file = out_dir.joinpath(f"sub-{sub_id}_roi-{roi_id}_desc-pearson_similarity.pkl")
            with open(out_file, "wb") as f:
                pickle.dump(gps, f)
