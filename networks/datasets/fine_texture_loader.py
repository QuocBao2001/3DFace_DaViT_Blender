import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os

import csv
from PIL import Image

class TextureDataset(Dataset):
    def __init__(self, csv_path, image_size, use_pose=True, transform=None):
        # define image transform to normalize input image
        if transform == None:
            self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        else:
            self.transform = transform

        # read data path from csv file
        self.data = []
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append(row)
                
        self.image_size = image_size
        self.use_pose = use_pose
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pcoarse_code, ptex_3dmm, ptex_face, p_tex_skin_seg, p_normal_uv = self.data[idx]

        #coarse_code = self.npy2labels(pcoarse_code)
        coarse_code = np.load(pcoarse_code)['arr_0']
        # Iterate over the keys in the loaded data
        # for key in coarse_code.keys():
        #     # Access the array using the key
        #     arr = coarse_code[key]
        #     # Print the loaded array
        #     print(f"Array with key '{key}': {arr}")

        tex_3dmm = Image.open(ptex_3dmm).convert('RGB')
        tex_face = Image.open(ptex_face).convert('RGB')
        # Load segmentation
        tex_skin_seg = self.load_mask(p_tex_skin_seg, self.image_size, self.image_size)
        normal_uv = np.load(p_normal_uv)['arr_0']


        # Apply transformations to the image
        tex_3dmm = self.transform(tex_3dmm)
        tex_face = self.transform(tex_face)

        data_dict = {
            'coarse_code': coarse_code,
            'tex_3dmm': tex_3dmm,
            'tex_face': tex_face,
            'tex_skin_seg': tex_skin_seg,
            'normal_uv': normal_uv,
        }

        if self.use_pose:
            angle_head_rotation = np.degrees(np.linalg.norm(coarse_code[150:153]))
            if 22.5 > angle_head_rotation and angle_head_rotation > -22.5:
                data_dict['good_pose'] = torch.BoolTensor([True])
            else:
                data_dict['good_pose'] = torch.BoolTensor([False])
        
        return data_dict
    
    def npy2labels(self, file_name):
        # get dictionary's labels
        dict_labels = np.load(file_name, allow_pickle=True).item()
                
        # reshape light params
        dict_labels['light_code'] = dict_labels['light_code'].reshape((-1,27))
        
        # dict keys in order: ['shape', 'exp', 'pose', 'texture_code', 'light_code', 'cam'])
        labels = np.concatenate([dict_labels['shape'], dict_labels['exp'], 
                                      dict_labels['pose'], dict_labels['texture_code'], 
                                      dict_labels['light_code'], dict_labels['cam']], axis=1)
        
        return labels
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)['arr_0']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask