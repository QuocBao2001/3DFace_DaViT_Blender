import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import csv
from PIL import Image


def get_image(image_or_path):
    image = io.imread(image_or_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 4:
        image = image[..., :3]
    return image

def plot(image, det, save_path='./example.png'):
    plt.imshow(image)
    plt.scatter(det[:, 0], det[:, 1], 5)
    plt.savefig(save_path)

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

    Lm3D = loadmat('/root/baonguyen/3d_face_reconstruction/focus_setup/BFM/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

# resize and crop images for face reconstruction
def resize_n_crop_img(w0, h0, lm, t, s, target_size=224., mask=None):
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return lm

# Open origin image file
image_path = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/images/Asian/m.0_1mf3t/000023_00@zh.jpg'
image = Image.open(image_path)

# Get the width and height of the image
width, height = image.size

index = 8

image = get_image('/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/Asian/m.0_1mf3t/seg_results/000023_00@zh_visualize.jpg')

landmark_file = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/Asian/m.0_1mf3t/list_landmarks.csv'

landmark_list = list(csv.reader(open(landmark_file),delimiter=','))

landmark_cpu = [int(x) for x in landmark_list[index][1:]]

raw_lm = np.reshape(np.asarray(landmark_cpu),(-1,2)).astype(np.float32)
raw_lm = np.stack([raw_lm[:,0], raw_lm[:,1]],1)

print(raw_lm)

if raw_lm.shape[0] != 5:
    lm5p = extract_5p(raw_lm)
else:
    lm5p = raw_lm

lm3D = load_lm3d()
# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
t, s = POS(lm5p.transpose(), lm3D.transpose())
rescale_factor=102.
s = rescale_factor/s

# processing the image
lm_new = resize_n_crop_img(width,height, raw_lm, t, s)

plot(image, lm_new)