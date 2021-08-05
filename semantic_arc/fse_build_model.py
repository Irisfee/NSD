# -*- coding: utf-8 -*-
"""
fse

Yufei Zhao
2020.3.3
"""

import gensim.downloader as api


import pandas as pd
import numpy as np
from fse import CSplitIndexedList
import re

from nltk import word_tokenize
from itertools import combinations 
import pickle 
from fse.models import SIF


glove = api.load("glove-wiki-gigaword-100")

# load caption files
captions = pd.read_csv('/projects/hulacon/shared/nsd_results/yufei/semantic_similarity/captions_by_picture.csv')
cap = captions.loc[:,['nsd_id', 'caption']].sort_values('nsd_id')

# remove punc
not_punc = re.compile('.*[A-Za-z0-9].*')

def prep_token(token):
    t = token.lower().strip("';.:()").strip('"')
    t = 'not' if t == "n't" else t
    return re.split(r'[-]', t)

def prep_sentence(sentence):
    tokens = []
    for token in word_tokenize(sentence):
        if not_punc.match(token):
            tokens = tokens + prep_token(token)
    return tokens


sentences = CSplitIndexedList(cap["caption"].tolist(), custom_split=prep_sentence)

# build model
model = SIF(glove, workers=28)
model.train(sentences)

# save object
filename = '/projects/hulacon/shared/nsd_results/yufei/semantic_similarity/fse_model'
outfile = open(filename,'wb')

pickle.dump(model, outfile)
outfile.close()