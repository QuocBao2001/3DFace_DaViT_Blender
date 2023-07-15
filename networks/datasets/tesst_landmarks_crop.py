import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt


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

image = get_image('/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/Asian/m.09gghx6/seg_results/000001_00@ja_visualize.jpg')

det = np.load('/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/Asian/m.09gghx6/seg_results/000001_00@ja_lmks.npy')

print(det.shape)

plot(image, det[48:68])