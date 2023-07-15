import os
import torch
import torch.nn
import numpy as np
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import pickle
import cv2

from utils import util
from utils.config import cfg
from coarse_model import C_model
from datasets.coarse_infer_loader import InferDataset

from utils.renderer import SRenderY, set_rasterizer
import torch.nn.functional as F

class CInfer():
    def __init__(self, config=None, texture_path=None):
        # load config
        if config==None:
            self.cfg = cfg
        else:
            self.cfg = config
        # set device, batch size and image size
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.batch_size = self.cfg.Cself.batch_size
        self.img_size = self.cfg.Cself.img_size
        # define model and load checkpoint
        self.model = C_model(config=self.cfg, device=self.device, use_fine_shape=True)

        self.load_checkpoint()

        # logs for watching
        logger.add(os.path.join(self.cfg.log_dir, 'infer.log'))

        if texture_path != None:  
            # dense mesh template, for save detail mesh
            self.texture_data = np.load(texture_path, allow_pickle=True, encoding='latin1').item()
    
    def load_checkpoint(self):
        # resume training, including model weight, steps
        if self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.checkpoint_path):
            checkpoint = torch.load(self.cfg.Cself.checkpoint_path)
            self.model.model.load_state_dict(checkpoint['state_dict'])
            # write log
            logger.info(f"resume training from {self.cfg.Cself.checkpoint_path}")
        elif not self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.pretrained_path):
            logger.info('Train from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Cself.pretrained_path)
            self.model.model.load_state_dict(pretrained_weight)
        elif self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.pretrained_path):
            logger.info('model path not found, start training from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Cself.pretrained_path)
        else:
            logger.info('can not find checkpoint or DaViT pretrained weight, training from scratch')

    def prepare_data(self):
        self.infer_dataset = InferDataset(self.cfg.Cself.csv_val_path, 
                                         image_size=self.cfg.dataset.image_size,
                                         scale=[self.cfg.dataset.scale_min, self.cfg.dataset.scale_max], 
                                         trans_scale=self.cfg.dataset.trans_scale)
        logger.info(f'---- infer data numbers: {len(self.infer_dataset)}')

        self.infer_dataloader = DataLoader(self.infer_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)
        self.infer_iter = iter(self.infer_dataloader)

    def fit(self):
        self.prepare_data()

        iter_through_data = int(np.ceil(len(self.infer_dataset)/self.batch_size))

        for step in tqdm(range(iter_through_data)):
            try:
                batch = next(self.infer_iter)
            except:
                self.infer_iter = iter(self.infer_dataloader)
                batch = next(self.infer_iter)

            self.model.model.eval()
            # get images, landmark and skin segmentation mask for currenct batch
            in_images = batch['image'].to(self.device)
            in_lmks = batch['landmark'].to(self.device)
            in_masks = batch['mask'].to(self.device)
            org_imgs = batch['org_img'].to(self.device)
            # infer code vector from images
            codedict, infer_code = self.model.encode(in_images)
            # change to dict
            #codedict = util.Ccode2dict(infer_code)
            codedict['images'] = org_imgs
            # get flame decode output dictionary
            opdict, visdict = self.model.decode(codedict, rendering = True, vis_lmk=False, return_vis=True)
        
            texture_imgs, texture_vis_masks = util.compute_texture_map(org_imgs, opdict['verts'], opdict['normals'], 
                                                                     codedict['cam'], self.texture_data)
            
            masks = in_masks.permute(0, 3, 1, 2)
            masks = self.closing_mask(masks).to(self.device)
            # print((masks*org_imgs).shape)
            texture_skin, texture_vis_skin = util.compute_texture_map(masks, opdict['verts'], opdict['normals'], 
                                                            codedict['cam'], self.texture_data)
            
            uv_skin_seg = texture_skin*texture_vis_skin
            #SHADING_UV = F.interpolate(visdict['shading_uv'],[256,256])
            #print(opdict['normals'].shape)
            for i in range(org_imgs.shape[0]):
                # full_head_tex = texture_imgs[i]
                # remove ocllusion cause by pose infor in texture_vis_masks

                coarse_code = infer_code[i].detach().cpu().numpy()
                tex_3dmm = opdict['albedo'][i]
                tex_face = texture_imgs[i] * texture_vis_masks[i]
                tex_skin_seg = uv_skin_seg[i].detach().cpu().numpy()
                normal_uv = visdict['uv_detail_normals'][i].detach().cpu().numpy()

                img_name = batch['path'][i] 

                print(img_name)

                #np.save(img_name.replace("_visualize.jpg", "_coarse_code.npy"), coarse_code)
                
                #np.save(img_name.replace("_visualize.jpg", "_tex_skin_seg.npy"), tex_skin_seg)

                np.savez_compressed(img_name.replace("_visualize.jpg", "_coarse_code.npz"), coarse_code)
                np.savez_compressed(img_name.replace("_visualize.jpg", "_tex_skin_seg.npz"), tex_skin_seg)
                np.savez_compressed(img_name.replace("_visualize.jpg", "_normal_uv.npz"), normal_uv)

                # with open(img_name.replace("_visualize.jpg", "_coarse_code.pkl"), "wb") as f:
                #     pickle.dump(coarse_code, f)
                # with open(img_name.replace("_visualize.jpg", "_tex_skin_seg.pkl"), "wb") as f:
                #     pickle.dump(tex_skin_seg, f)
                # with open(img_name.replace("_visualize.jpg", "_normal_uv.pkl"), "wb") as f:
                #     pickle.dump(normal_uv, f)

                # Check if the file exists
                # dict_name = ["_coarse_code.pkl", "_tex_skin_seg.pkl", "_normal_uv.pkl"]
                # for file_tail in dict_name:
                #     filename = img_name.replace("_visualize.jpg", file_tail)
                #     if os.path.exists(filename):
                #         # Delete the file
                #         os.remove(filename)
                #         print(f"{filename} deleted.")

                util.save_tensor_img_cv(tex_3dmm, img_name.replace("_visualize.jpg", "_tex_3dmm.jpg"))
                util.save_tensor_img_cv(tex_face, img_name.replace("_visualize.jpg", "_tex_face.jpg"))
                # util.save_tensor_img_cv(torch.cat([uv_skin_seg[i],uv_skin_seg[i],uv_skin_seg[i]], dim=0), './visualize_tex_skin_seg.jpg')
                # util.save_tensor_img_cv(visdict['uv_detail_normals'][i], './visualize_normal_fine.jpg')
                # util.save_tensor_img_cv(visdict['normal_uv'][i], img_name.replace("_visualize.jpg", "_normal_uv.jpg"))
                # util.save_tensor_img_cv(torch.cat([uv_skin_seg[i],uv_skin_seg[i],uv_skin_seg[i]], dim=0), img_name.replace("_visualize.jpg", "_tex_skin_seg.jpg"))

    def closing_mask(self, tensor_image):
        # Define the closing kernel size
        kernel_size = 9

        # Convert the tensor images to numpy arrays
        images_np = tensor_image.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (batch_size, height, width, channels)

        # Perform morphological closing on each image in the batch
        closed_images_np = []
        for img_np in images_np:
            # Convert the image to grayscale
            # img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Perform morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed_img_gray = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)

            # Add the closed image to the list
            closed_images_np.append(closed_img_gray)

        # Convert the list of closed images back to a PyTorch tensor
        closed_images = torch.from_numpy(np.stack(closed_images_np)).unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        return closed_images