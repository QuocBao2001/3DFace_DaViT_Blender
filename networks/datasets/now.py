import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = '/root/baonguyen/3d_face_reconstruction/evaluate/dataset/NoW_Dataset'
        self.data_path = os.path.join(folder, 'file_paths.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.crop_size = crop_size
        self.scale = scale
            
    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_path = bbx_path.replace("iphone_pictures", "detected_face")
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']; right = bbx_data['right']
        top = bbx_data['top']; bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:,:,:3]

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(old_size*self.scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2,0,1)

        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        tensor_img = transform(torch.tensor(dst_image).float())

        return {'image': tensor_img,
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                'origin_image': torch.tensor(dst_image).float()
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
    
class NoWCompareNeural(Dataset):
    def __init__(self, resultfolder):
        folder = '/root/baonguyen/3d_face_reconstruction/evaluate/dataset/NoW_Dataset'
        self.data_path = os.path.join(folder, 'file_paths.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.resultfolder = resultfolder
            
    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        #folder_path = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'

        item_path = os.path.join(self.resultfolder, self.data_lines[index].strip().split('.')[0])
        item_name =  item_path.split('/')[-1].split('.')[0]
        item_path = os.path.dirname(item_path)
        person_ids = os.path.basename(os.path.dirname(item_path))
        recognition_data = np.load(os.path.join(item_path, item_name + "_output_regcognition_our.npz"))
        input = torch.from_numpy(recognition_data['target'])
        _3dmm = torch.from_numpy(recognition_data['_3dmm'])
        _3dmm_align = torch.from_numpy(recognition_data['_3dmm_align'])
        face_pixel = torch.from_numpy(recognition_data['face_pixel'])
        face_pixel_align = torch.from_numpy(recognition_data['face_pixel_align'])
        fine_tex = torch.from_numpy(recognition_data['fine_tex'])
        fine_tex_align = torch.from_numpy(recognition_data['fine_tex_align'])
        
        return_dict = {
            'imagename': item_path,
            'person_ids': person_ids,
            'input': input,
            '_3dmm': _3dmm,
            '_3dmm_align': _3dmm_align,
            'face_pixel': face_pixel,
            'face_pixel_align': face_pixel_align,
            'fine_tex': fine_tex,
            'fine_tex_align': fine_tex_align,
        }


        return return_dict