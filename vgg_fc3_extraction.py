from pathlib import Path
import h5py
from torchvision import models
from PIL import Image
from torchvision import transforms
import numpy as np
from torch import nn
import argparse
from itertools import product
from scipy import spatial, stats
import simplejson as json
import pandas as pd
from collections import OrderedDict

import pickle

# Get parameters
parser = argparse.ArgumentParser(description="Parameters.")
parser.add_argument(
    "-s",
    "--sub_id",
    action="store",
    default=None,
    type=str,
    help="Subject identifier (the sub- prefix should be removed)."
)

## subject info
args = parser.parse_args()
sub_id = args.sub_id

## VGG fc layer features
# import stimuli
sti_dir = Path('/projects/hulacon/shared/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5').as_posix()
sti = h5py.File(sti_dir,'r')
sti_array = sti['imgBrick']


#######################
#  Model preparation  #
#######################
# mini-batches of 3-channel RGB images of shape (3 x H x W)
# define preprocess parameters
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_preprocess(img):
    """
    Preprocess the image and create the tensor
    """
    im = Image.fromarray(img)
    input_tensor = preprocess(im) # create tensor
    input_batch = input_tensor.unsqueeze(0)# create a mini-batch as expected by the model
    return input_batch

class VggPartial(nn.Module):
    """
    forward vgg network till the selected layer
    """
    supported_layers = ['fc1', 'relu1', 'fc2', 'relu2', 'fc3']

    def __init__(self,
                 layer):
        super(VggPartial, self).__init__()
        assert layer in VggPartial.supported_layers
        self.model = models.vgg16(pretrained=True)
        self.output_layer = layer
        keep_upto = {
            'fc1': 0,
            'relu1': 1,
            'fc2': 3,
            'relu2': 4,
            'fc3': 6}[layer]
        classifier = list(
            self.model.classifier.children())[:keep_upto + 1]
        self.model.classifier = nn.Sequential(*classifier)
        
    def forward(self, x):
        return self.model.forward(x)

# create all the models
selected_layer = 'fc3'

model = VggPartial(layer = selected_layer)
model.eval()


## beh file
beh_dir = Path('/projects/hulacon/shared/nsd_results/yufei/codes_yufei/content/beh')
res_dir = beh_dir.joinpath(f'sub-{sub_id}_beh.tsv')
beh = pd.read_csv(res_dir, sep = '\t')

fc_features = []
for iImage in beh['73KID']:
    img = sti_array[(iImage-1),:,:,:]
    input_batch = image_preprocess(img)
    current_features = model(input_batch).data.numpy()
    fc_features.append(np.squeeze(current_features))
    
fc_features = np.array(fc_features)

# outdir
out_dir = Path('/projects/hulacon/shared/nsd_results/yufei/codes_yufei/content/fc')
out_dir.mkdir(exist_ok=True, parents=True)
out_fid = out_dir.joinpath(
            f'sub-{sub_id}_fc.pkl')

open_file = open(out_fid, "wb")
pickle.dump(fc_features, open_file)
open_file.close()