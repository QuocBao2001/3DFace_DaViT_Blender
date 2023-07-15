import os
import torch
import torch.nn
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import util
from utils.config import cfg
from utils.loss import CodeLoss
from coarse_model import C_model
from datasets.GIF_synthetic import GIFDataset

class CSupTrain():
    def __init__(self, config=None):
        # load config
        if config==None:
            self.cfg = cfg
        else:
            self.cfg = config
        # set device, batch size and image size
        self.device = self.cfg.device + ':' + self.cfg.device_id
        self.batch_size = self.cfg.Csup.batch_size
        self.img_size = self.cfg.Csup.img_size
        # define model and optimizer, load checkpoint
        self.model = C_model(config=self.cfg)
        self.configure_optimizers()
        self.load_checkpoint()

        # define loss function
        self.weights = torch.Tensor([self.cfg.Csup.loss.shape_w , self.cfg.Csup.loss.exp_w, 
                                     self.cfg.Csup.loss.pose_w, self.cfg.Csup.loss.tex_w, 
                                     self.cfg.Csup.loss.light_w, self.cfg.Csup.loss.cam_w]).to(self.device)
        self.L2_loss = CodeLoss(weights=self.weights).to(self.device)

        # logs for watching
        logger.add(os.path.join(self.cfg.log_dir, 'train.log'))
        self.train_losses = []
        self.val_losses = []

        # save best eval model loss
        self.best_val_loss = 99999

        # make dir if not exist
        os.makedirs(self.cfg.coarse_model_dir, exist_ok=True)
        os.makedirs(self.cfg.flame.vis_dir, exist_ok=True)

    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
                                self.model.model.parameters(),
                                lr=self.cfg.Csup.lr,)
    
    def load_checkpoint(self):
        # resume training, including model weight, opt, steps
        if self.cfg.Csup.resume and os.path.exists(self.cfg.Csup.checkpoint_path):
            checkpoint = torch.load(self.cfg.Csup.checkpoint_path)
            self.model.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['opt'])
            #util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            # write log
            logger.info(f"resume training from {self.cfg.Csup.checkpoint_path}")
            logger.info(f"training start from step {self.global_step}")
        elif not self.cfg.Csup.resume and os.path.exists(self.cfg.Csup.pretrained_path):
            logger.info('Train from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Csup.pretrained_path)
            self.model.model.load_state_dict(pretrained_weight)
            self.global_step = 0
        elif self.cfg.Csup.resume and os.path.exists(self.cfg.Csup.pretrained_path):
            logger.info('model path not found, start training from DaViT pretrained')
            pretrained_weight = torch.load(self.cfg.Csup.pretrained_path)
            self.global_step = 0
        else:
            logger.info('can not find checkpoint or DaViT pretrained weight, training from scratch')
            self.global_step = 0

    def prepare_data(self):
        self.train_dataset = GIFDataset(self.cfg.Csup.train_dir)
        self.val_dataset = GIFDataset(self.cfg.Csup.val_dir)
        logger.info(f'---- training data numbers: {len(self.train_dataset)}')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.Csup.num_worker,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=False,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
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
            if self.global_step % self.cfg.Csup.checkpoint_steps*10 == 0:
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

                images = batch[0].to(self.device)
                target_code = batch[1].to(self.device)
                
                #print(images.shape[0])

                with torch.no_grad():
                    infer_code = self.model.model(images)
                    loss += self.L2_loss(infer_code,target_code)
            
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

        for epoch in range(start_epoch, self.cfg.Csup.num_epochs):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.Csup.num_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                self.model.model.train()
                images = batch[0].to(self.device)
                target_code = batch[1].to(self.device)
                infer_code = self.model.model(images)
                loss = self.L2_loss(infer_code, target_code)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.global_step % self.cfg.Csup.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
                    # for k, v in losses.items():
                    #     loss_info = loss_info + f'{k}: {v:.4f}, '
                        # if self.cfg.Csup.write_summary:
                        #     self.writer.add_scalar('train_loss/total_loss',loss, global_step=self.global_step)                    
                    loss_info = loss_info + f', total loss: {loss:.4f}, '
                    logger.info(loss_info)
                    self.train_losses.append((loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.Csup.vis_steps == 0:
                    # visual infer
                    codedict = util.Ccode2dict(infer_code)
                    codedict['images'] = batch[2].to(self.device)
                    visind = list(range(8))

                    opdict, visdict = self.model.decode(codedict)

                    if 'predicted_images' in opdict.keys():
                        visdict['predicted_images'] = opdict['predicted_images'][visind]

                    savepath = os.path.join(self.cfg.flame.vis_dir, f'{self.global_step:06}.jpg')
                    self.model.visualize_grid(visdict, savepath=savepath, size=224, dim=1, return_gird=True)

                    # visual target
                    codedict = util.Ccode2dict(target_code)
                    codedict['images'] = batch[2].to(self.device)
                    visind = list(range(8))

                    opdict, visdict = self.model.decode(codedict)

                    if 'predicted_images' in opdict.keys():
                        visdict['predicted_images'] = opdict['predicted_images'][visind]

                    savepath = os.path.join(self.cfg.flame.vis_dir, f'{self.global_step:06}_target.jpg')
                    self.model.visualize_grid(visdict, savepath=savepath, size=224, dim=1, return_gird=True)


                if self.global_step>0 and self.global_step % self.cfg.Csup.checkpoint_steps == 0:
                    self.save_model()

                if self.global_step % self.cfg.Csup.val_steps == 0:
                    val_loss = self.validation_step()
                    self.val_losses.append((val_loss.detach().cpu().numpy(), self.global_step))

                if self.global_step % self.cfg.Csup.plot_steps == 0:
                    self.visualize_train_process()
                
                # if self.global_step % self.cfg.Csup.eval_steps == 0:
                #     self.evaluate()

                self.global_step += 1
                if self.global_step > self.cfg.Csup.num_steps:
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