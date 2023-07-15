import os
import torch
import scipy
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from torch.utils.data import Dataset
from skimage.transform import estimate_transform, warp, resize, rescale
from torchvision import transforms
from PIL import Image
import torchvision


from datasets import detectors

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        # elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
        #     self.imagepath_list = video2sequence(testpath, sample_step)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

        self.transform_input = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        img_PIL = Image.open(imagepath).convert('RGB')
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
            kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        
        tensor_img = self.transform_input(torch.tensor(dst_image).float())

        return {'image': tensor_img,
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(dst_image).float(),
                }
    
    # def __getitem__(self, index):
    #     imagepath = self.imagepath_list[index]
    #     imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
    #     image = np.array(imread(imagepath))
    #     if len(image.shape) == 2:
    #         image = image[:,:,None].repeat(1,1,3)
    #     if len(image.shape) == 3 and image.shape[2] > 3:
    #         image = image[:,:,:3]

    #     h, w, _ = image.shape
    #     print(self.iscrop)
    #     if self.iscrop:
    #         # provide kpt as txt file, or mat file (for AFLW2000)
    #         kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
    #         kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
    #         if os.path.exists(kpt_matpath):
    #             print('branch1')
    #             kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
    #             left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    #             top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
    #             old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
    #         elif os.path.exists(kpt_txtpath):
    #             print('branch2')
    #             kpt = np.loadtxt(kpt_txtpath)
    #             left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    #             top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
    #             old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
    #         else:
    #             print('branch3')
    #             bbox, bbox_type = self.face_detector.run(image)
    #             print(bbox)
    #             if len(bbox) < 4:
    #                 print('no face detected! run original image')
    #                 left = 0; right = h-1; top=0; bottom=w-1
    #             else:
    #                 left = bbox[0]; right=bbox[2]
    #                 top = bbox[1]; bottom=bbox[3]
    #             old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
    #         size = int(old_size*self.scale)
    #         src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    #     else:
    #         src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
    #     DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
    #     tform = estimate_transform('similarity', src_pts, DST_PTS)
        
    #     image = image/255.

    #     dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
    #     dst_image = dst_image.transpose(2,0,1)
        
    #     # save cropped_img
    #     to_PIL = torchvision.transforms.ToPILImage()
    #     _img = to_PIL(torch.tensor(dst_image).float()[0])
    #     _img.save('/root/baonguyen/3d_face_reconstruction/output/27_05_only_real/test/1.jpg')
    #     return {'image': torch.tensor(dst_image).float(),
    #             'imagename': imagename,
    #             'tform': torch.tensor(tform.params).float(),
    #             'original_image': torch.tensor(image.transpose(2,0,1)).float(),
    #             }
