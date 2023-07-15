import os
import torch
import torch.nn
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.io import imread
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np

from utils import util
from utils.masks_generate import generate_mask
from utils.config import cfg
from utils.loss import CodeLoss
#from utils.loss import IDMRFLoss
from models.Texture_fine_net import Fine_net
from datasets.fine_texture_loader import TextureDataset

class FTextureTrain():
    def __init__(self, config=None):
        self.global_step = 0
        # load config
        if config==None:
            self.cfg = cfg
        else:
            self.cfg = config
        # set device, batch size and image size
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.batch_size = self.cfg.finetex.batch_size
        self.img_size = self.cfg.finetex.img_size
        # define model and optimizer, load checkpoint
        self.model = Fine_net().to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        self.memory_save = False
        #self.mrf_loss = IDMRFLoss()
        


        # logs for watching
        logger.add(os.path.join(self.cfg.log_dir, 'train.log'))
        self.train_losses = []
        self.val_losses = []

        # save best eval model loss
        self.best_val_loss = 99999

        # make dir if not exist
        os.makedirs(self.cfg.fine_model_dir, exist_ok=True)
        os.makedirs(self.cfg.flame.vis_dir, exist_ok=True)

        # masks for loss
        mask_wthout_eyes = imread(self.cfg.flame.face_wthout_eye).astype(np.float32)/255.
        mask_wthout_eyes = self.process_mask(mask_wthout_eyes,0.5)
        self.mask_wthout_eyes = torch.from_numpy(mask_wthout_eyes).permute(2,0,1).to(self.device)

        # Load eyes mask loss
        eye_right = imread(self.cfg.flame.mask_right_eye_path).astype(np.float32)/255.
        eye_right = self.process_mask(eye_right,0.4)
        self.mask_right_eye = torch.from_numpy(eye_right).permute(2,0,1).to(self.device)
        

        eye_left = imread(self.cfg.flame.mask_left_eye_path).astype(np.float32)/255.
        eye_left = self.process_mask(eye_left,0.4)
        self.mask_left_eye = torch.from_numpy(eye_left).permute(2,0,1).to(self.device)
    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
                                self.model.parameters(),
                                lr=self.cfg.finetex.lr,)
    
    def load_checkpoint(self):
        # resume training, including model weight, opt, steps
        if self.cfg.finetex.resume and os.path.exists(self.cfg.finetex.checkpoint_path):
            checkpoint = torch.load(self.cfg.finetex.checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['opt'])
            #util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            # write log
            logger.info(f"resume training from {self.cfg.finetex.checkpoint_path}")
            logger.info(f"training start from step {self.global_step}")
        elif not self.cfg.finetex.resume and os.path.exists(self.cfg.finetex.pretrained_path):
            logger.info('Train from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.finetex.pretrained_path)
            self.model.load_state_dict(pretrained_weight['state_dict'])
            self.global_step = 0
        elif self.cfg.finetex.resume and os.path.exists(self.cfg.finetex.pretrained_path):
            logger.info('model path not found, start training from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.finetex.pretrained_path)
            self.global_step = 0
        else:
            logger.info('can not find checkpoint or DaViT pretrained weight, training from scratch')
            self.global_step = 0

    def prepare_data(self):
        self.train_dataset = TextureDataset(self.cfg.finetex.csv_train, self.cfg.finetex.img_size)
        self.val_dataset = TextureDataset(self.cfg.finetex.csv_val, self.cfg.finetex.img_size)
        logger.info(f'---- training data numbers: {len(self.train_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.finetex.num_worker,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
        self.val_iter = iter(self.val_dataloader)
    
    def save_model(self, is_best=False):
        model_dict = {}
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['opt'] = self.opt.state_dict()
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size
        if is_best:
            torch.save(model_dict, os.path.join(cfg.fine_model_dir, f'best_at_{self.global_step:08}.tar'))  
        else:
            torch.save(model_dict, os.path.join(cfg.fine_model_dir, 'model' + '.tar'))   
            # 
            if self.global_step % self.cfg.finetex.checkpoint_steps*10 == 0:
                if self.memory_save == False:
                    os.makedirs(os.path.join(cfg.fine_model_dir), exist_ok=True)
                    torch.save(model_dict, os.path.join(cfg.fine_model_dir, f'{self.global_step:08}.tar'))   
                    self.memory_save = True
                else:
                    self.memory_save = False

    def validation_step(self):
            self.model.eval()
            loss = 0
            num_iters = int(len(self.val_dataset)/self.batch_size)
            for step in tqdm(range(num_iters), desc=f"Val iter"):
                try:
                    batch = next(self.val_iter)
                except:
                    self.val_iter = iter(self.val_dataloader)
                    batch = next(self.val_iter)

                tex_3dmm = batch['tex_3dmm'].to(self.device)
                tex_face = batch['tex_face'].to(self.device)
                tex_skin_seg =  batch['tex_skin_seg'].to(self.device)
                # good_pose = batch['good_pose'].unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)
                normal_uv = batch['normal_uv'].to(self.device)
                coarse_code = batch['coarse_code'].to(self.device)

                input_model = torch.cat([tex_3dmm, tex_face], dim=1)

                with torch.no_grad():
                    infer_tex = self.model(input_model)

                    infer_tex_lighting = self.lighting_uv_tex(infer_tex, normal_uv, coarse_code[:, 206:233])

                    tex_3dmm_lighting = self.lighting_uv_tex(tex_3dmm, normal_uv, coarse_code[:, 206:233])

                    tex_face_remove_lights = self.remove_lighting_uv_tex(tex_face, normal_uv, coarse_code[:, 206:233])

                    loss_dict = self.compute_loss(infer_tex_lighting, tex_face, 
                                              tex_3dmm_lighting, tex_3dmm, infer_tex, tex_skin_seg,
                                              tex_face_remove_lights)

                    loss += loss_dict['total_loss']
                    
                    loss_info = f"Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    for k, v in loss_dict.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '                   
                    logger.info(loss_info)
                    self.train_losses.append((loss.detach().cpu().numpy(), self.global_step))
                
            
            loss = loss / num_iters
            logger.info(f'validate loss: {loss:.4f}')
            if loss < self.best_val_loss:
                self.save_model(is_best=True)   
                self.best_val_loss = loss
            self.model.train()
            return loss

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch

        for epoch in range(start_epoch, self.cfg.finetex.num_epochs):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.finetex.num_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                self.model.train()
                tex_3dmm = batch['tex_3dmm'].to(self.device)
                tex_face = batch['tex_face'].to(self.device)
                tex_skin_seg =  batch['tex_skin_seg'].to(self.device)
                #good_pose = batch['good_pose'].unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)
                normal_uv = batch['normal_uv'].to(self.device)
                coarse_code = batch['coarse_code'].to(self.device)

                # make more oclussion for good pose texture
                #tex_override = tex_face.clone()
                # random get input 
                # tex_override = torch.empty_like(tex_face)
                # tex_override[:,] = torch.where(good_pose[:,]==True, self.override_2_tex(tex_face[:,], tex_face[0]),tex_face[:,])
                # tex_override = tex_override.to(self.device)

                tex_override = self.override_2_tex(tex_face[:,], tex_face[0]).to(self.device)
                # for i in range(self.batch_size):
                #     if good_pose[i] == True:
                #         cv2.imshow('tex override', tex_override[i].permute(1,2,0).detach().cpu().numpy())
                #         cv2.imshow('tex ', tex_face[i].permute(1,2,0).detach().cpu().numpy())
                #         cv2.imshow('tex 0', tex_face[0].permute(1,2,0).detach().cpu().numpy())
                #         cv2.waitKey(0)

                # mask, rect = generate_mask('stroke', [256,256], [256,256])
                # self.mask_stroke = torch.from_numpy(mask).to(self.device).repeat([self.batch_size, 1, 1, 1])
                # tex_override = tex_face*self.mask_stroke
                input_model = torch.cat([tex_3dmm, tex_override], dim=1)

                infer_tex = self.model(input_model)

                infer_tex_lighting = self.lighting_uv_tex(infer_tex, normal_uv, coarse_code[:, 206:233])

                tex_3dmm_lighting = self.lighting_uv_tex(tex_3dmm, normal_uv, coarse_code[:, 206:233])

                tex_face_remove_lights = self.remove_lighting_uv_tex(tex_face, normal_uv, coarse_code[:, 206:233])

                loss_dict = self.compute_loss(infer_tex_lighting, tex_face, tex_3dmm_lighting, tex_3dmm, infer_tex, tex_skin_seg, tex_face_remove_lights)
                loss = loss_dict['total_loss']
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.global_step % self.cfg.finetex.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    for k, v in loss_dict.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '                   
                    logger.info(loss_info)
                    self.train_losses.append((loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.finetex.vis_steps == 0:
                    # visual infer
                    visdict = {}
                    visdict['tex_3dmm'] = tex_3dmm
                    visdict['tex_face'] = tex_face
                    visdict['tex_override'] = tex_override
                    visdict['tex_face_remove_lights'] = tex_face_remove_lights
                    visdict['skin_Seg'] = tex_skin_seg
                    visdict['infer_tex'] = infer_tex*(1-tex_skin_seg)
                    visdict['infer_tex_lighting'] = infer_tex_lighting
                    savepath = os.path.join(self.cfg.finetex.vis_dir, f'{self.global_step:06}.jpg')
                    self.visualize_grid(visdict, savepath)

                if self.global_step>0 and self.global_step % self.cfg.finetex.checkpoint_steps == 0:
                    self.save_model()

                if self.global_step % self.cfg.finetex.val_steps == 0:
                    val_loss = self.validation_step()
                    self.val_losses.append((val_loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.finetex.plot_steps == 0:
                    self.visualize_train_process()
                
                # if self.global_step % self.cfg.finetex.eval_steps == 0:
                #     self.evaluate()

                self.global_step += 1
                if self.global_step > self.cfg.finetex.num_steps:
                    break

    def visualize_train_process(self):
        # Plot the training and validation losses
        train_losses_values, train_losses_iterations = zip(*self.train_losses)
        val_losses_values, val_losses_iterations = zip(*self.val_losses)
        plt.plot(train_losses_iterations, train_losses_values, label='Train Loss')
        plt.plot(val_losses_iterations, val_losses_values, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Save the plot to an image file
        plt.savefig(os.path.join(self.cfg.flame.vis_dir, 'loss_plot.png'))

        # Clear the current figure
        plt.clf()

    def compute_loss(self, infer_tex_lighting, tex_face, tex_3dmm_light, tex_3dmm, infer_tex, masks, tex_face_remove_lights):

        skin_wout_eye = self.mask_wthout_eyes*masks
        #----------- loss to make model use fine detail from image
        #tex_face_loss = (masks*(infer_tex_lighting - tex_face).abs()).mean()
        tex_face_loss = (skin_wout_eye*(infer_tex_lighting - tex_face).abs()).mean()

        #dmm_loss = ((1-masks)*(infer_tex_lighting - tex_3dmm_light).abs()).mean()
        #----------- loss to ensure that the output not too different with 3dmm
        #dmm_loss = (infer_tex_lighting - tex_3dmm_light).abs().mean()
        # count the mean different between infer texture and input 3dmm texture in skin region
        differ_3dmm = infer_tex - tex_3dmm
        differ_3dmm_mean = (skin_wout_eye*(differ_3dmm)).mean()
        
        # the infer_tex in non-skin region must be equal 3dmm + differ_3dmm
        transfer_3dmm_loss = (self.mask_wthout_eyes*(1-masks)*((infer_tex - differ_3dmm_mean) - tex_3dmm)).abs().mean()

        #----------- loss use to reconstruct region if only have face show in image
        #flip_differ_3dmm = torch.flip(differ_3dmm, [3])
        # flip_infer_tex = torch.flip(infer_tex, [3])
        # vertical_sym_loss = (self.mask_wthout_eyes*(1-masks)*(infer_tex - flip_infer_tex).abs()).mean()

        flip_tex_face_rm_light = torch.flip(masks*tex_face_remove_lights, [3])
        masks_flip = torch.flip(masks, [3])
        #vertical_sym_loss = (self.mask_wthout_eyes*masks_flip*(1-masks)*(infer_tex - flip_tex_face_rm_light).abs()).mean()

        vertical_sym_loss = (self.mask_wthout_eyes*masks_flip*(masks)*(infer_tex - flip_tex_face_rm_light).abs()).mean()

        #vertical_sym_loss = (self.mask_wthout_eyes*(1-masks)*(differ_3dmm - flip_differ_3dmm).abs()).sum()

        # util.save_tensor_img_cv(differ_3dmm[0], "diff.jpg")
        # util.save_tensor_img_cv(flip_differ_3dmm[0], "diff_flip.jpg")

        #----------- loss to erase boundary in image
        # get edge in skin uv mask
        edge_mask_i = torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])
        edge_mask_j = torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])

        # Apply dilation to make edge thicker
        size_kernel = self.cfg.finetex.bound_thickness
        num_pad = size_kernel//2
        dilated_edge_i = F.max_pool2d(edge_mask_i, kernel_size=size_kernel, stride=1, padding=num_pad)
        dilated_edge_j = F.max_pool2d(edge_mask_j, kernel_size=size_kernel, stride=1, padding=num_pad)

        # compute the different between near pixel in edge region
        diff_i = (torch.abs(infer_tex_lighting[:, :, 1:, :] - infer_tex_lighting[:, :, :-1, :]))*dilated_edge_i
        diff_j = (torch.abs(infer_tex_lighting[:, :, :, 1:] - infer_tex_lighting[:, :, :, :-1]))*dilated_edge_j
        bound_loss = torch.mean(diff_i) + torch.mean(diff_j)

        # # Define the desired output size
        # smallersize = (128, 128)  # Specify the width and height

        # # Resize the image using interpolate
        # resized_image = F.interpolate(infer_tex_lighting, size=smallersize, mode='bilinear', align_corners=False)
        # resized_image = resized_image

        # # compute the different between near pixel in edge region
        # diff_i = (torch.abs(resized_image[:, :, 1:, :] - resized_image[:, :, :-1, :]))
        # diff_j = (torch.abs(resized_image[:, :, :, 1:] - resized_image[:, :, :, :-1]))
        # bound_loss = torch.mean(diff_i) + torch.mean(diff_j)

        # eyes loss
        eyes_loss = self.compute_eyes_loss(infer_tex_lighting, tex_3dmm_light, tex_face, masks)

        # mrf loss
        #detail_mrf_loss = self.mrf_loss(infer_tex_lighting*masks, tex_face*masks)

        # + dmm_loss*self.cfg.finetex.tex_3dmm_loss
        total_loss = tex_face_loss*self.cfg.finetex.face_loss  + transfer_3dmm_loss*self.cfg.finetex.transfer_3dmm_loss +\
                     vertical_sym_loss*self.cfg.finetex.sym_loss + bound_loss*self.cfg.finetex.bound_loss +\
                          eyes_loss*self.cfg.finetex.eyes_loss #+ detail_mrf_loss*self.cfg.finetex.mrfloss
        loss = {
            'tex_face_loss': tex_face_loss,
            #'dmm_loss': dmm_loss,
            'transfer_3dmm_loss': transfer_3dmm_loss,
            'vertical_sym_loss': vertical_sym_loss,
            'bound_loss': bound_loss,
            'eyes_loss':eyes_loss,
            #'mrf_loss': detail_mrf_loss,
            'total_loss': total_loss
        }
        return loss

    def lighting_uv_tex(self, infer_tex, normal_uv, sh_coeff):
        sh_coeff = sh_coeff.reshape(-1, 9, 3)
        shading = self.add_SHlight(normal_uv, sh_coeff)
        return infer_tex * shading

    def remove_lighting_uv_tex(self, tex, normal_uv, sh_coeff):
        sh_coeff = sh_coeff.reshape(-1, 9, 3)
        shading = self.add_SHlight(normal_uv, sh_coeff)
        #shading = torch.where(shading == 0., 1)
        return tex / (shading)

    def visualize_grid(self, visdict, savepath=None, size=256, dim=1, return_gird=True):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim==2
        grids = {}
        for key in visdict:
            _,_,h,w = visdict[key].shape
            if dim == 2:
                new_h = size; new_w = int(w*size/h)
            elif dim == 1:
                new_h = int(h*size/w); new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu(),
                                                     nrow=self.cfg.finetex.batch_size)
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath:
            cv2.imwrite(savepath, grid_image)
        if return_gird:
            return grid_image
    
    def override_2_tex(self, batch_org_tex, mask_tex):
        mask_tex = torch.sum(mask_tex, dim=0)
        return torch.where(mask_tex[:,:]<=0.05,0,batch_org_tex[:,:,:])
    
    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        
        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                           ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                           (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float().to(self.device)

        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1).to(self.device) # [bz, 9, h, w]
        sh = sh*constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading
    
    def process_mask(self, np_mask, threshold):
        mask = np.zeros_like(np_mask)
        # for i in range(1, 16):
        mask[np_mask>threshold] = 1.
        mask = mask[:,:,0:1]
        return mask
    
    def compute_eyes_loss(self, infer_tex_lighting, tex_3dmm_light, tex_face, masks):
        batch_s =  masks.shape[0]
        ER_loss = 0
        EL_loss = 0
        for i in range(batch_s):
            
            ER_mask = self.mask_right_eye*masks[i]
            EL_mask = self.mask_left_eye*masks[i]
            
            if torch.sum(ER_mask)/self.cfg.finetex.pixel_per_eye < cfg.finetex.ratio_occlution:
                ER_loss += (self.mask_right_eye*(infer_tex_lighting[i] - tex_3dmm_light[i]).abs()).mean()
                
            else: 
                ER_loss += (self.mask_right_eye*(infer_tex_lighting[i] - tex_face[i]).abs()).mean()

            if torch.sum(EL_mask)/self.cfg.finetex.pixel_per_eye < cfg.finetex.ratio_occlution:
                EL_loss += (self.mask_left_eye*(infer_tex_lighting[i] - tex_3dmm_light[i]).abs()).mean()
                
            else: 
                EL_loss += (self.mask_left_eye*(infer_tex_lighting[i] - tex_face[i]).abs()).mean()

        return (ER_loss + EL_loss)/batch_s