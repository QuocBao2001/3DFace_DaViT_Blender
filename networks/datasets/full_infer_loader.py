import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset
import os

import csv
from PIL import Image

class InferDataset(Dataset):
    def __init__(self, csv_path, image_size, scale, trans_scale = 0, transform=None):
        # define image transform to normalize input image
        if transform == None:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        else:
            self.transform = transform

        self.to_Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        # read data path from csv file
        self.data = []
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)\
                
        self.image_size = image_size
        # scale
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        
        # Load image
        image = imread(img_path)/255.
        # crop information
        tform = self.crop(image, kpt)
        ## crop 
        cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        
        # normalized kpt
        cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

        tensor_img = self.transform(cropped_image)
    
        data_dict = {
            'image': tensor_img.float(),
            'landmark': torch.from_numpy(cropped_kpt).float(),
            'mask': torch.from_numpy(cropped_mask).float(),
            'org_img': self.to_Tensor(cropped_image).float(),
            'path': img_path,
        }
        
        return data_dict

    def crop(self, image, kpt):
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

            h, w, _ = image.shape
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
            # translate center
            trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
            center = center + trans_scale*old_size # 0.5

            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
            size = int(old_size*scale)

            # crop image
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            
            return tform