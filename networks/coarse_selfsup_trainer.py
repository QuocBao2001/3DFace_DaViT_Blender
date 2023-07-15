import os
import torch
import torch.nn
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import util
from utils import loss
from utils.config import cfg
from coarse_model import C_model
from datasets.coarse_self_supervise_loader import BUPTDataset

class CSelfTrain():
    def __init__(self, config=None):
        # load config
        if config==None:
            self.cfg = cfg
        else:
            self.cfg = config
        # set device, batch size and image size
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.batch_size = self.cfg.Cself.batch_size
        self.img_size = self.cfg.Cself.img_size
        # define model and optimizer, load checkpoint
        self.model = C_model(config=self.cfg)
        self.configure_optimizers()
        self.load_checkpoint()

        # logs for watching
        logger.add(os.path.join(self.cfg.log_dir, 'train.log'))
        self.train_losses = []
        self.val_losses = []

        # save best eval model loss
        self.best_val_loss = 99999

        # make dir if not exist
        os.makedirs(self.cfg.coarse_model_dir, exist_ok=True)
        os.makedirs(self.cfg.flame.vis_dir, exist_ok=True)

        self.id_loss = loss.VGGFace2Loss(pretrained_model=self.cfg.Cself.fr_model_path)     

    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
                                self.model.model.parameters(),
                                lr=self.cfg.Cself.lr,)
    
    def load_checkpoint(self):
        # resume training, including model weight, opt, steps
        if self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.checkpoint_path):
            checkpoint = torch.load(self.cfg.Cself.checkpoint_path)
            self.model.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['opt'])
            #util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            # write log
            logger.info(f"resume training from {self.cfg.Cself.checkpoint_path}")
            logger.info(f"training start from step {self.global_step}")
        elif not self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.pretrained_path):
            logger.info('Train from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Cself.pretrained_path)
            self.model.model.load_state_dict(pretrained_weight['state_dict'])
            self.global_step = 0
        elif self.cfg.Cself.resume and os.path.exists(self.cfg.Cself.pretrained_path):
            logger.info('model path not found, start training from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Cself.pretrained_path)
            self.global_step = 0
        else:
            logger.info('can not find checkpoint or DaViT pretrained weight, training from scratch')
            self.global_step = 0

    def prepare_data(self):
        self.train_dataset = BUPTDataset(self.cfg.Cself.csv_train_path, 
                                         image_size=self.cfg.dataset.image_size,
                                         scale=[self.cfg.dataset.scale_min, self.cfg.dataset.scale_max], 
                                         trans_scale=self.cfg.dataset.trans_scale)
        
        self.val_dataset = BUPTDataset(self.cfg.Cself.csv_val_path,
                                       image_size=self.cfg.dataset.image_size,
                                        scale=[self.cfg.dataset.scale_min, self.cfg.dataset.scale_max], 
                                        trans_scale=self.cfg.dataset.trans_scale)
        
        logger.info(f'---- training data numbers: {len(self.train_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.Cself.num_worker,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)
        self.val_iter = iter(self.val_dataloader)
    
    def save_model(self, is_best=False):
        model_dict = {}
        model_dict['state_dict'] = self.model.model.state_dict()
        model_dict['opt'] = self.opt.state_dict()
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size
        if is_best:
            torch.save(model_dict, os.path.join(cfg.coarse_model_dir, f'best_at_{self.global_step:08}.tar'))  
        else:
            torch.save(model_dict, os.path.join(cfg.coarse_model_dir, 'model' + '.tar'))   
            # 
            if self.global_step % self.cfg.Cself.checkpoint_steps*10 == 0:
                os.makedirs(os.path.join(cfg.coarse_model_dir), exist_ok=True)
                torch.save(model_dict, os.path.join(cfg.coarse_model_dir, f'{self.global_step:08}.tar'))   

    def validation_step(self):
            self.model.model.eval()
            loss = 0
            num_iters = int(len(self.val_dataset)/self.batch_size)
            for step in tqdm(range(num_iters), desc=f"Val iter"):
                try:
                    batch = next(self.val_iter)
                except:
                    self.val_iter = iter(self.val_dataloader)
                    batch = next(self.val_iter)

                # get images, landmark and skin segmentation mask for currenct batch
                in_images = batch['image'].to(self.device)
                in_lmks = batch['landmark'].to(self.device)
                in_masks = batch['mask'].to(self.device)

                with torch.no_grad():
                    # infer code vector from images
                    infer_code = self.model.model(in_images)
                    # change to dict
                    codedict = util.Ccode2dict(infer_code)
                    codedict['images'] = in_images
                    # get flame decode output dictionary
                    opdict = self.model.decode(codedict, rendering = True, vis_lmk=False, return_vis=False)
                    
                    loss += self.compute_loss(in_images, in_lmks,  in_masks, opdict, codedict)['all_loss']
            
            loss = loss / num_iters
            logger.info(f'validate loss: {loss:.4f}')
            if loss < self.best_val_loss:
                self.save_model(is_best=True)   
                self.best_val_loss = loss
            self.model.model.train()
            return loss

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch

        # epoch loop
        for epoch in range(start_epoch, self.cfg.Cself.num_epochs):
            # step in each epoch
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.Cself.num_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                # change model to train mode
                self.model.model.train()
                # get images, landmark and skin segmentation mask for currenct batch
                in_images = batch['image'].to(self.device)
                in_lmks = batch['landmark'].to(self.device)
                in_masks = batch['mask'].to(self.device)
                org_img = batch['org_img'].to(self.device)
                # infer code vector from images
                infer_code = self.model.model(in_images)
                # change to dict
                codedict = util.Ccode2dict(infer_code)
                codedict['images'] = in_images
                # get flame decode output dictionary
                opdict = self.model.decode(codedict, rendering = True, vis_lmk=False, return_vis=False)
                
                # compute loss
                losses = self.compute_loss(org_img, in_lmks,  in_masks, opdict, codedict)
                all_loss = losses['all_loss']
                self.opt.zero_grad()
                all_loss.backward()
                self.opt.step()

                if self.global_step % self.cfg.Cself.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '                   
                    loss_info = loss_info + f', total loss: {all_loss:.4f}, '
                    logger.info(loss_info)
                    self.train_losses.append((all_loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.Cself.vis_steps == 0:
                    # visual infer
                    codedict = util.Ccode2dict(infer_code)
                    codedict['images'] = batch['org_img'].to(self.device)
                    visind = list(range(8))

                    opdict, visdict = self.model.decode(codedict)

                    if 'predicted_images' in opdict.keys():
                        visdict['predicted_images'] = opdict['predicted_images'][visind]

                    savepath = os.path.join(self.cfg.flame.vis_dir, f'{self.global_step:06}.jpg')
                    self.model.visualize_grid(visdict, savepath=savepath, size=224, dim=1, return_gird=True)


                if self.global_step>0 and self.global_step % self.cfg.Cself.checkpoint_steps == 0:
                    self.save_model()

                if self.global_step % self.cfg.Cself.val_steps == 0:
                    val_loss = self.validation_step()
                    self.val_losses.append((val_loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.Cself.plot_steps == 0:
                    self.visualize_train_process()
                
                # if self.global_step % self.cfg.Cself.eval_steps == 0:
                #     self.evaluate()

                self.global_step += 1
                if self.global_step > self.cfg.Cself.num_steps:
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

    def compute_loss(self, org_img, in_lmks,  in_masks, opdict, codedict):
        losses = {}
        if self.cfg.Cself.loss.photo > 0.:
            #------ rendering
            # mask
            mask_face_eye = F.grid_sample(self.model.uv_face_eye_mask.expand(self.batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
            # images
            predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
            opdict['predicted_images'] = predicted_images
        
        # landmark loss
        predicted_landmarks = opdict['landmarks2d']
        if self.cfg.Cself.loss.useWlmk:
            losses['landmark'] = loss.weighted_landmark_loss(predicted_landmarks, in_lmks)*self.cfg.Cself.loss.lmk
        else:    
            losses['landmark'] = loss.landmark_loss(predicted_landmarks, in_lmks)*self.cfg.Cself.loss.lmk
        
        # photometric loss
        if self.cfg.Cself.loss.photo > 0.:
            if self.cfg.Cself.loss.useSeg:
                masks = in_masks.permute(0, 3, 1, 2)
                masks = masks[:,None,:,:]
            else:
                masks = mask_face_eye*opdict['alpha_images']
            losses['photometric_texture'] = (masks*(predicted_images - org_img).abs()).mean()*self.cfg.Cself.loss.photo

        #identity loss
        if self.cfg.Cself.loss.id > 0.:
            shading_images = self.model.render.add_SHlight(opdict['normal_images'], codedict['lights'].detach())
            albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
            overlay = albedo_images*shading_images*mask_face_eye + org_img*(1-mask_face_eye)
            losses['identity'] = self.id_loss(overlay, org_img) * self.cfg.Cself.loss.id
        
        # regularization loss
        losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.Cself.loss.reg_shape
        losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.Cself.loss.reg_exp
        losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.Cself.loss.reg_tex
        losses['light_reg'] = ((torch.mean(codedict['lights'], dim=2)[:,:,None] - codedict['lights'])**2).mean()*self.cfg.Cself.loss.reg_light
        
        # sum all loss
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses
        

