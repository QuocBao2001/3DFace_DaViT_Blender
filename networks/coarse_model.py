import os
import cv2
import torch
import numpy as np
import torchvision
import torch.nn as nn
from skimage.io import imread
import torch.nn.functional as F

from utils import util
from utils.config import cfg
#from datasets import datasets
from models.FLAME import FLAME, FLAMETex
from models.Encoder_net import DaViT_Encoder 
from utils.tensor_cropper import transform_points
from utils.renderer import SRenderY, set_rasterizer
from models.encoders_fine_shape import ResnetEncoder
from models.decoders_fine_shape import Generator
from time import time

torch.backends.cudnn.benchmark = True

class C_model(nn.Module):
    def __init__(self, config=None, device='cuda', use_fine_shape=False):
        super(C_model, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.img_size = self.cfg.Csup.img_size
        self.use_fine_shape = use_fine_shape

        self._create_model()

        if self.use_fine_shape:
            self._create_fine_shape()

        self._setup_renderer(self.cfg.flame)
    
    def _setup_renderer(self, model_cfg):
        set_rasterizer()
        self.render = SRenderY(self.img_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size, rasterizer_type=model_cfg.rasterizer_type).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self):
        # define coarse model
        self.model = DaViT_Encoder(output_dims=self.cfg.flame.total_params).to(self.device)

        # define flame model for visualize
        self.flame = FLAME(self.cfg.flame).to(self.device)
        #if self.cfg.flame.use_tex:
        self.flametex = FLAMETex(self.cfg.flame).to(self.device)

        if self.cfg.isInfer:
            state_dict_path = self.cfg.Cself.state_dict_path
            if os.path.exists(state_dict_path):
                print(f'trained model found. load {state_dict_path}')
                checkpoint = torch.load(state_dict_path)
                self.model.load_state_dict(checkpoint)
        else:
            # resume model
            model_path = self.cfg.Cself.checkpoint_path
            if os.path.exists(model_path):
                print(f'trained model found. load {model_path}')
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                print(f'cannot found model.')
        self.model.eval()

    def _create_fine_shape(self):
        self.E_fine_shape = ResnetEncoder(outsize=128).to(self.device)
        self.D_fine_shape = Generator(latent_dim=128+53, out_channels=1, out_scale=0.01, sample_mode = 'bilinear').to(self.device)
        
        # resume model
        fine_shape_model_path = self.cfg.fine_shape_model_path
        if os.path.exists(fine_shape_model_path):
            print(f'trained model found. load {fine_shape_model_path}')
            checkpoint = torch.load(fine_shape_model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_fine_shape.state_dict(), checkpoint['E_fine_shape'])
            util.copy_state_dict(self.D_fine_shape.state_dict(), checkpoint['D_fine_shape'])
        else:
            print(f'please check model path: {fine_shape_model_path}')
        
        self.E_fine_shape.eval()
        self.D_fine_shape.eval()

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    # @torch.no_grad()
    def encode(self, images):
        start_time = time()
        parameters = self.model(images)
        end_time = time()
        infer_time = end_time - start_time
        self.cfg.count_time += infer_time
        codedict = parameters
        # parameters = self.model(images)
        # codedict = util.Ccode2dict(parameters)
        # codedict['images'] = images 

        # if self.use_fine_shape:
        #     detailcode = self.E_fine_shape(images)
        #     codedict['detail'] = detailcode
        
        return codedict, parameters

    def decode_albedo(self, codedict):
        albedo = self.flametex(codedict['tex'])
        return albedo

    # @torch.no_grad()
    def decode(self, codedict, rendering=True, vis_lmk=True, return_vis=True,
                render_orig=False, original_image=None, tform=None, fine_tex=None):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        if self.cfg.flame.use_tex:
            albedo = self.flametex(codedict['tex'])
        elif fine_tex != None:
            albedo = fine_tex
        else:
            albedo = self.flametex(codedict['tex'])
            #albedo = torch.zeros([batch_size, 3, self.cfg.flame.uv_size, self.cfg.flame.uv_size], device=images.device) 
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.img_size/2 + self.img_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.img_size/2 + self.img_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.img_size, self.img_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.img_size, self.img_size
            background = None

        if rendering:
            lighting = codedict['lights']
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            if self.cfg.flame.use_tex:
                ops = self.render(verts, trans_verts, albedo, lights=lighting, h=h, w=w, background=background)
            elif fine_tex==None :
                ops = self.render(verts, trans_verts, albedo, lights=lighting, h=h, w=w, background=background)
            else:
                ops = self.render(verts, trans_verts, albedo, h=h, w=w, background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['shading_images'] = ops['shading_images']
        
        #if self.cfg.flame.use_tex:
            opdict['albedo'] = albedo
            opdict['normals'] = ops['normals']
        
        if self.use_fine_shape:
            uv_z = self.D_fine_shape(torch.cat([codedict['pose'][:,3:], codedict['exp'], codedict['detail']], dim=1))
            uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['lights'])
            uv_texture = albedo*uv_shading

            opdict['uv_texture'] = uv_texture 
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z+self.fixed_uv_dis[None,None,:,:]
            
            # upsample and fine tune 3d mesh
            dense_vertices_list = []
            for i in range(opdict['verts'].shape[0]):
                dense_vertices, dense_faces = util.upsample_mesh(opdict['verts'][i].cpu().numpy(),
                                                                                opdict['normals'][i].cpu().numpy(), 
                                                                                self.render.faces[0][i].cpu().numpy(), 
                                                                                opdict['displacement_map'][i].cpu().numpy().squeeze(), 
                                                                                opdict['albedo'][i].cpu().numpy(), 
                                                                                self.dense_template)
                dense_vertices_list.append(dense_vertices)
            
            dense_vertices_list = [torch.from_numpy(arr) for arr in dense_vertices_list]

            # Stack the tensors along a new dimension to create a batch
            dense_vertices_list = torch.stack(dense_vertices_list, dim=0).to(self.device)
            
            trans_dense_verts = dense_vertices_list
            #trans_dense_verts = util.batch_orth_proj(dense_vertices_list, codedict['cam'])
            #rans_dense_verts[:,:,1:] = -trans_dense_verts[:,:,1:]

            if return_vis and render_orig and original_image is not None and tform is not None:
                trans_dense_verts = transform_points(trans_dense_verts, tform, points_scale, [h, w])
            # self.render_fine = SRenderY(self.img_size, obj_filename=self.cfg.flame.topology_path, 
            #                             uv_size=self.cfg.flame.uv_size, rasterizer_type=self.cfg.flame.rasterizer_type).to(self.device)
            #ops_fine = self.render(dense_vertices, trans_dense_verts, opdict['albedo'], h=h, w=w, background=background)
            opdict['dense_vertices'] = trans_dense_verts

        if vis_lmk:
            normals68 = self.flame.seletec_3d68(ops['transformed_normals'])
            landmarks3d_vis = (normals68[:,:,2:] < 0.1).float()
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            ## render shape
            shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=background, return_grid=True)

            uv_normal = self.render.world2uv(ops['normals'])

            visdict = {
                #'inputs': images, 
                'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                'shape_images': shape_images,
                'normal_uv': uv_normal,
            }
            if self.use_fine_shape:
                detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)*alpha_images
                shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images, h=h, w=w, images=background)
                visdict['shape_detail_images'] = shape_detail_images
                visdict['uv_detail_normals'] =  uv_detail_normals

            visdict['inputs']= images
            #if self.cfg.flame.use_tex:
            visdict['rendered_images'] = ops['images']
            

            return opdict, visdict

        else:
            return opdict
    
    def visualize_grid(self, visdict, savepath=None, size=224, dim=1, return_gird=True):
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
            
            grtemp=F.interpolate(visdict[key], [new_h, new_w])

            grids[key] = torchvision.utils.make_grid(grtemp.detach().cpu())
        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        if savepath:
            cv2.imwrite(savepath, grid_image)
        if return_gird:
            return grid_image
    
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices, faces, 
                        texture=texture, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vertices, 
                        dense_faces,
                        colors = dense_colors,
                        inverse_face_order=True)
    
    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1.-self.uv_face_eye_mask)
        return uv_detail_normals

    # def run(self, imagepath, iscrop=True):
    #     ''' An api for running deca given an image path
    #     '''
    #     testdata = datasets.TestData(imagepath)
    #     images = testdata[0]['image'].to(self.device)[None,...]
    #     codedict = self.encode(images)
    #     opdict, visdict = self.decode(codedict)
    #     return codedict, opdict, visdict