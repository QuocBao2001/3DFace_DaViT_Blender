import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import os
from utils.config import cfg

#labels = np.load("C:/thesis_repo/3d_face_reconstruction/Dataset/GIF_generated/params.npy", allow_pickle=True).item()

labels = np.load(os.path.join(cfg.Csup.val_dir, 'params.npy'), allow_pickle=True).item()

# Set the printing options
# np.set_printoptions(threshold=np.inf)


# Print the entire array
# with np.printoptions(threshold=np.inf):
#     print(labels[0])

print(labels.keys())

for key in labels.keys():
    print(labels[key].shape)

print(labels['identity_indices'][50])

# reshape light params
labels['light_code'] = labels['light_code'].reshape((-1,27))

labels = np.concatenate([labels['shape'], labels['exp'], labels['pose'],
                         labels['texture_code'], labels['light_code'], labels['cam']], axis=1)

print(labels.shape)