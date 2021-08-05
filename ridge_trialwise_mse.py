import os
from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from io_func.project import Project
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection 
import math
import random
from itertools import product
from sklearn.model_selection import permutation_test_score

# Get parameters
parser = argparse.ArgumentParser(description="Parameters.")
parser.add_argument(
    "--sub_id",
    action="store",
    default=None,
    type=str,
    help="Subject identifier (the sub- prefix should be removed)."
)

proj = Project()

# directory
deriv_dir = proj.deriv_dir
wk_dir = deriv_dir.joinpath("decode")
beh_dir = wk_dir.joinpath('beh')
roi_dir = proj.roi_dir
fc_dir = wk_dir.joinpath('fc')


def trial_wise_mse(
    model, training_feature, testing_feature, training_label, testing_label
):
    # fit the model
    model.fit(training_feature, training_label)
    pred = model.predict(testing_feature)
    mse = np.mean((testing_label-pred)**2, axis=1)
    return mse


def permutation_list(iteration, range_num=10000):
    seed_list = random.sample(range(1, range_num), iteration)
    if len(set(seed_list)) != iteration:  # make sure seeds are unique
        raise TypeError("Seed duplication.")
    return seed_list


# Read ROI definition
roi_spec = proj.read_roi_spec(roi_dir.joinpath('roi_definition_parcel.yaml'))
# Read ROI list
roi_info = proj.read_roi_list(roi_dir.joinpath('roi_list_parcel.yaml'))

# parameters
comp = 10
args = parser.parse_args()
sub_id = args.sub_id
ses_list = proj.ses_list[sub_id]
iteration = 100

# read in beh file
beh = pd.read_csv(beh_dir.joinpath(f'sub-{sub_id}_beh.tsv'), sep='\t')

# read in fc file
fc_fid = fc_dir.joinpath(f'sub-{sub_id}_fc.pkl')
open_file = open(fc_fid, "rb")
fc = pickle.load(open_file)
open_file.close()

# reduce fc into desired components
pca = PCA(n_components=comp)
fc_reduced = pca.fit_transform(fc)
pca.explained_variance_ratio_.sum()

mse_all = []
# loop through roi
for roi_id in roi_info:
    # read in roi img
    # ROI space
    space = roi_info[roi_id]["space"]
    # Make ROI mask
    roi_mask = proj.make_roi_from_spec(sub_id, roi_id, roi_spec, space=space)
    # Read beta data
    beta_data = proj.read_singletrial_beta_roi(
        sub_id,
        roi_mask,
        ses_list=ses_list,
        space=space,
        file_type="hdf5",
    )
    # remove any col that has nan
    beta_data = beta_data[:, ~np.isnan(beta_data).any(axis=0)]
    # first rep - sec rep
    beta_dif = beta_data[beh['i1_trial_num'],:] - beta_data[beh['i2_trial_num'],:]

    # 10-fold cross validation with ridge regression
    kf_10 = model_selection.KFold(
        n_splits=10,
        shuffle=True,
        random_state=int(sub_id))

    # ridge regression model
    model_ridge = Ridge(alpha=1, normalize=True)
    
    res = []
    for train_index, test_index in kf_10.split(range(beta_dif.shape[0])):
        # feature
        training_feature = beta_dif[train_index, :]
        testing_feature = beta_dif[test_index, :]
        # label
        training_label = fc_reduced[train_index, :]
        testing_label = fc_reduced[test_index, :]

        # get mse
        mse = trial_wise_mse(
            model_ridge, training_feature, testing_feature, training_label, testing_label
        )

        # permutation seed list
        seed_list = permutation_list(iteration)
        # permutation
        permute_mse = []
        for seed in seed_list:
            # shuffle training label
            permute_training_label = shuffle(training_label, random_state=seed)
            # get permutated mse
            permute_mse_current = trial_wise_mse(
                model_ridge, training_feature, testing_feature, permute_training_label, testing_label
            )
            permute_mse.append(permute_mse_current)
        # create a dataframe for permutation results
        permute_mse = np.array(permute_mse)
        permute_mse_t = np.transpose(permute_mse)
        col_name = ['per_mse_' + str(i) for i in range(iteration)]
        permute_mse_df = pd.DataFrame(permute_mse_t, columns=col_name)

        # cv results
        cv_res = pd.DataFrame({
                    'roi_id': roi_id,
                    'trial_id': test_index,
                    'mse': mse,
                    'permute_mse': np.mean(permute_mse, axis=0)
                })
        cv_res_all = pd.concat([cv_res, permute_mse_df], axis=1)

        res.append(cv_res_all)
    res = pd.concat(res)
    res = res.sort_values(by=['trial_id']).reset_index(drop=True)
    res_mse = beh.merge(res, left_index=True, right_index=True)
    mse_all.append(res_mse)
mse_all = pd.concat(mse_all)

# outdir
out_dir = wk_dir.joinpath('res')
out_dir.mkdir(exist_ok=True, parents=True)
out_fid = out_dir.joinpath(
            f'sub-{sub_id}_ridge_100_pc-10_normalized_trialwise-mse.tsv')
mse_all.to_csv(out_fid, sep='\t', float_format='%.5f', na_rep='n/a', index=False)
