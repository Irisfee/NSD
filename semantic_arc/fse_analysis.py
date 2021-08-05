import pandas as pd
import numpy as np
import re

from nltk import word_tokenize
from itertools import combinations 
import pickle 
from fse.models import SIF

filename = '/projects/hulacon/shared/nsd_results/yufei/semantic_similarity/fse_model'
infile = open(filename,'rb')
model = pickle.load(infile)
infile.close()
import os

# Directories
bids_dir = '/projects/hulacon/shared/nsd'
data_dir = '/projects/hulacon/shared/nsd/nsddata'
beh_dir ='/projects/hulacon/shared/nsd/nsddata/ppdata'
code_dir = '/projects/hulacon/shared/nsd_results/yufei'
sem_dir = '/projects/hulacon/shared/nsd_results/yufei/semantic_similarity'

# subject info
subj_list = ["01", "02", "03", "04", "05", "06", "07", "08"]
ses_list = ["40", "40", "32", "30", "40", "32", "40", "30"]

# Loop for subjects
for subj_id, n_ses in zip(subj_list, ses_list):
    
    # create output dir
    out_dir = os.path.join(sem_dir,f'sub-{subj_id}')
    
    # read in all beh data
    dat = []   
    dat = pd.read_csv(
           os.path.join(beh_dir, f'subj{subj_id}', 'behav', 'responses.tsv'),
            sep='\t')
    
    # Session list
    session_list = ["%.2d" % (i+1) for i in range(int(n_ses))]
                          
    # loop for session
    for session_id in session_list:
        out_file = os.path.join(out_dir, f'sub-{subj_id}_ses-{session_id}_semantic_similarity_fse.tsv')
        
        beh = dat[dat['SESSION'] == int(session_id)]
        nsd_id = {'nsd_id': beh['73KID'].unique()}
        ses_cap_id = pd.DataFrame(nsd_id)
        
        semantic_similarity_fse = []
        for sent_1, sent_2 in [(sent_1, sent_2) for sent_1 in ses_cap_id.nsd_id for sent_2 in ses_cap_id.nsd_id]:
            p = model.sv.similarity(sent_1 - 1, sent_2 - 1)
            semantic_similarity_fse.append({'picture_1': sent_1,
                                              'picture_2': sent_2,
                                              'similarity': p})

        semantic_similarity_fse = pd.DataFrame(semantic_similarity_fse)
        semantic_similarity_fse.to_csv(out_file, sep='\t', index=None, float_format='%.6f', na_rep='n/a')