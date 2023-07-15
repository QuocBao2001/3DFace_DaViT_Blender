import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import util
from utils.config import cfg
from coarse_model import C_model
from models.Texture_fine_net import Fine_net
from datasets.now import NoWDataset

def add_SHlight(normal_images, sh_coeff):
    '''
        sh_coeff: [bz, 9, 3]
    '''
    
    ## SH factors for lighting
    pi = np.pi
    constant_factor = torch.tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
                        ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
                        (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).float().to(cfg.device)

    N = normal_images
    sh = torch.stack([
            N[:,0]*0.+1., N[:,0], N[:,1], \
            N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
            N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
            ], 
            1).to(cfg.device) # [bz, 9, h, w]
    sh = sh*constant_factor[None,:,None,None]
    shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
    return shading

def lighting_uv_tex(infer_tex, normal_uv, sh_coeff):
    sh_coeff = sh_coeff.reshape(-1, 9, 3)
    shading = add_SHlight(normal_uv, sh_coeff)
    return infer_tex * shading

def evaluate():
    ''' NOW validation 
    '''
    os.makedirs(os.path.join(cfg.output_dir, 'NOW_validation'), exist_ok=True)
    savefolder = os.path.join(cfg.output_dir, 'NOW_validation', f'') 
    os.makedirs(savefolder, exist_ok=True)

    # run C_model
    device = cfg.device
    model = C_model(config = cfg, device=device,  use_fine_shape=True)
    model.eval()

    # run skin_fine model
    skin_model = Fine_net().to(device)
    checkpoint = torch.load(cfg.finetex.checkpoint_path)
    skin_model.load_state_dict(checkpoint['state_dict'])
    skin_model.eval()

    # run now validation images
    dataset = NoWDataset(scale=(cfg.dataset.scale_min + cfg.dataset.scale_max)/2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=False)
    faces = model.flame.faces_tensor.cpu().numpy()

    dense_infor = np.load(cfg.flame.dense_infor_path)

    dense_faces = dense_infor['dense_face']

    uv_dense_vertices = dense_infor['uv_coords']

    # get texture data
    texture_data = np.load(cfg.flame.dense_template_path, allow_pickle=True, encoding='latin1').item()

    for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
        images = batch['image'].to(device)
        org_image = batch['origin_image'].to(device)
        imagename = batch['imagename']
        with torch.no_grad():
            # encode image to coarse code
            codedict, parameters = model.encode(images)
            codedict['images'] = org_image

            # decode albedo
            tex_3dmm = model.decode_albedo(codedict)

            # decode coarse code
            #_, visdict = model.decode(codedict)

                        # get flame decode output dictionary
            opdict, visdict_0 = model.decode(codedict, rendering = True, vis_lmk=False, return_vis=True)
        
            texture_imgs, texture_vis_masks = util.compute_texture_map(org_image, opdict['verts'], opdict['normals'], 
                                                                     codedict['cam'], texture_data)
            
            tex_face = texture_imgs * texture_vis_masks

            input_fine_tex_model = torch.cat([tex_3dmm, tex_face], dim=1)

            fine_texture = skin_model(input_fine_tex_model)

            fine_texture_lighting = lighting_uv_tex(fine_texture, visdict_0['uv_detail_normals'], codedict['lights'])
            
            tex_3dmm_lighting = lighting_uv_tex(tex_3dmm, visdict_0['normal_uv'], codedict['lights'])

            tex_3dmm_lighting_fine_normal = lighting_uv_tex(tex_3dmm, visdict_0['uv_detail_normals'], codedict['lights'])
            tex_3dmm_lighting_fine_normal = lighting_uv_tex(fine_texture, visdict_0['uv_detail_normals'], codedict['lights'])

            # decode with fine texture have lighting
            opdict, visdict = model.decode(codedict, fine_tex=fine_texture_lighting)

            # decode with 3dmm coarse texture to compare
            opdict, visdict_3 = model.decode(codedict, fine_tex=tex_face)

            codedict['exp'][:] = 0.
            codedict['pose'][:] = 0.
            codedict['cam'][:, 1:] = 0.

            opdict, visdict2 = model.decode(codedict, fine_tex=fine_texture_lighting)
        #-- save results for evaluation
        verts = opdict['verts'].cpu().numpy()
        landmark_51 = opdict['landmarks3d_world'][:, 17:]
        landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
        landmark_7 = landmark_7.cpu().numpy()
        for k in range(images.shape[0]):
            os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
            # save mesh
            util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
            # save 7 landmarks for alignment
            np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
            for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
                #print(vis_name)
                if vis_name == 'normal_uv':
                    continue
                # import ipdb; ipdb.set_trace()
                image = util.tensor2image(visdict[vis_name][k])
                name = imagename[k].split('/')[-1]
                # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
                cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
        # visualize results to check
        # visdict.pop('normal_uv')

        #-- save fine shape result
        dense_vertices = opdict['dense_vertices'].cpu().numpy()
        fine_texture_np = fine_texture_lighting.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        print(fine_texture_np.shape)
        for k in range(images.shape[0]):
            # save mesh
            util.write_obj(os.path.join(savefolder, f'{imagename[k]}_fine.obj'),
                          vertices=dense_vertices[k],
                          faces=dense_faces,
                          colors=None,
                          texture=fine_texture_np[k][:, :, ::-1],
                          uvcoords=uv_dense_vertices,
                          uvfaces=dense_faces,
                          inverse_face_order=False,
                          normal_map=None,
                         )      

        visdict.pop('normal_uv')
        visdict["shape_align"] = visdict2['shape_images']
        visdict["rendered_align"] = visdict2['rendered_images']
        visdict["tex_from_img"] = visdict_3['rendered_images']
        visdict["tex_face"] = texture_imgs
        visdict["tex_face_visualize"] = tex_face
        visdict["tex_3dmm"] = tex_3dmm_lighting
        visdict["3dmm_tex"] = visdict_0['rendered_images']
        visdict["fine_texture"] = fine_texture
        visdict["tex_lighting"] = fine_texture_lighting

        util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))

evaluate()