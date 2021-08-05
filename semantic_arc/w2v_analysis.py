# -*- coding: utf-8 -*-
"""
word2vec similarity analysis
session number and subj_list need to be change for future analysis
parallel computing performed over each subject each session

Yufei Zhao
2019.8.21
"""

# basic setting
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# alway import pyemd before gensim to avoid the bug
from pyemd import emd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import argparse
import multiprocessing as mp

# Get parallelization parameters
parser = argparse.ArgumentParser(description="Parallelization parameters.")
parser.add_argument(
    "--n_procs",
    action="store",
    default=1,
    type=int,
    help="The maximum number of processes running in parallel.."
)

parser.add_argument(
    '--participant_label',
    action='store',
    nargs='+',
    help='One or more participant identifiers (the sub- prefix should be removed).')

parser.add_argument(
    "--n_ses",
    action="store",
    type=int,
    help="The maximum number of processes running in parallel.."
)

args = parser.parse_args()
n_procs = int(args.n_procs)
subj_list = args.participant_label
n_ses = int(args.n_ses)


# Directories
from pathlib import Path
bids_dir = Path('/projects/hulacon/shared/nsd')
data_dir = bids_dir.joinpath('nsddata')
beh_dir = data_dir.joinpath('ppdata')

code_dir = Path('/projects/hulacon/shared/nsd_results/yufei')
sem_dir = code_dir.joinpath('semantic_similarity')

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.

# Remove stopwords.
stop_words = stopwords.words('english')

# load google pre-trained model
model = KeyedVectors.load_word2vec_format(sem_dir.joinpath('GoogleNews-vectors-negative300.bin'), binary=True)


# Get participants list
# subj_info = pd.read_csv(code_dir.joinpath('participants.tsv'), delimiter='\t')
# subj_list = [i.replace('sub-', '') for i in subj_info['participant_id'].tolist()]

# subj_list = ['01', '02']

# Session list
session_list = ["%.2d" % (i+1) for i in range(n_ses)]

# Read in captions
captions = pd.read_csv(sem_dir.joinpath('captions_by_picture.csv'))
cap = captions.loc[:,['nsd_id', 'caption']]

# Pre-processing a document.
from nltk import word_tokenize
download('punkt')  # Download data for tokenizer.

def _preprocess(doc):
    """
    Preprocess the doc.
    Lower each word. Split into words. Remove stopwords, numbers, adn punctuation.
    """
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc

# Calculate the mean vector
def _get_mean_vector(words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in model.vocab]
    if len(words) >= 1:
        return np.mean(model[words], axis=0)
    else:
        return []

# get cosine distance (dot product)
def _cos_simi(vec1, vec2):
    similarity = np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return similarity

# Calculate similarity
def _cal_w2v(ses_cap, out_file):
    
    # preprocess each paragraph
    ses_cap = ses_cap.assign(prep = ses_cap.caption.apply(lambda x: _preprocess(x)))    

    # get the averaged word vector for each picture
    ses_cap = ses_cap.assign(avg_vec = ses_cap.prep.apply(lambda x: _get_mean_vector(x)) )
    
    # map through all the sentences combinations to calculate the similarity
    semantic_similarity_w2v = []

    for sent_1, sent_2 in [(sent_1, sent_2) for sent_1 in ses_cap.nsd_id for sent_2 in ses_cap.nsd_id]:
        indx_1 = ses_cap.index[ses_cap['nsd_id'] == sent_1].tolist()[0]
        indx_2 = ses_cap.index[ses_cap['nsd_id'] == sent_2].tolist()[0]
        p = _cos_simi(ses_cap.avg_vec[indx_1],ses_cap.avg_vec[indx_2])
        semantic_similarity_w2v.append({'picture_1': sent_1,
                                        'picture_2': sent_2,
                                        'similarity': p})

    semantic_similarity_w2v = pd.DataFrame(semantic_similarity_w2v)
    semantic_similarity_w2v.to_csv(out_file, sep='\t', index=None, float_format='%.6f', na_rep='n/a')


# Loop for subjects
pool = mp.Pool(processes=n_procs)

for subj_id in subj_list:
    
    # create output dir
    out_dir = sem_dir.joinpath(f'sub-{subj_id}')
    out_dir.mkdir(exist_ok=True, parents=True)
      
    # read in all beh data
    dat = []   
    dat = pd.read_csv(
           beh_dir.joinpath(f'subj{subj_id}', 'behav',
                              'responses.tsv'),
            sep='\t')
    
    # loop for session
    for session_id in session_list:

        out_file = out_dir.joinpath(
                    f'sub-{subj_id}_ses-{session_id}_'
                    f'semantic_similarity_w2v.tsv')
        
        # filter out the trials within the session
        beh = dat[dat['SESSION'] == int(session_id)]
        
        # get unique captions in the session
        nsd_id = {'nsd_id': beh['73KID'].unique()}
        ses_cap_id = pd.DataFrame(nsd_id)
        ses_cap = pd.merge(ses_cap_id, cap, on='nsd_id', how='left')

        pool.apply_async(_cal_w2v,(ses_cap, out_file))

pool.close()
pool.join()
