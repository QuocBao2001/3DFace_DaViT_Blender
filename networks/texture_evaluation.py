import os
import cv2
import torch
import csv
import copy
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from utils import util
from utils.config import cfg
from coarse_model import C_model
from models.Texture_fine_net import Fine_net
from datasets.now import NoWDataset
from utils import loss

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

def fill_flip(texface):
    flipped_images = torch.flip(texface, [3])

    # Convert black pixels in the original image to a binary mask
    black_pixels = (texface <= 0.05).all(dim=1)
    black_pixels_expanded = black_pixels.unsqueeze(1).expand_as(flipped_images)

    # Use broadcasting to fill the corresponding pixels in the flipped image with black color
    filled_image_tensor = torch.where(black_pixels_expanded, flipped_images, texface)

    return filled_image_tensor


def evaluate(id_loss, writer):
    ''' NOW validation 
    '''
    os.makedirs(os.path.join(cfg.output_dir, 'NOW_validation'), exist_ok=True)
    savefolder = os.path.join(cfg.output_dir, 'NOW_validation', f'') 
    os.makedirs(savefolder, exist_ok=True)

    # load C_model
    device = cfg.device
    model = C_model(config = cfg, device=device,  use_fine_shape=True)
    model.eval()

    # load skin_fine model
    skin_model = Fine_net().to(device)
    checkpoint = torch.load(cfg.finetex.checkpoint_path)
    skin_model.load_state_dict(checkpoint['state_dict'])
    skin_model.eval()

    # load now validation images
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

    # loop through batch images
    for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
        images = batch['image'].to(device)
        org_image = batch['origin_image'].to(device)
        imagename = batch['imagename']
        
        with torch.no_grad():
            # encode image to coarse code
            codedict, infer_code = model.encode(images)
            codedict['images'] = org_image

            # get codedict with pose zeros and camera zeros
            code_dict_align = copy.deepcopy(codedict)
            code_dict_align['pose'][:] = 0.
            code_dict_align['cam'][:, 1:] = 0.

            # # get codedict with pose pi/12 and camera zeros
            # code_dict_angle1= copy.deepcopy(codedict)
            # code_dict_angle1['pose'][:] = 0.
            # code_dict_angle1['pose'][:,1] = np.pi/12
            # code_dict_angle1['cam'][:, 1:] = 0.
            # # get codedict with pose -pi/12 and camera zeros
            # code_dict_angle2= copy.deepcopy(codedict)
            # code_dict_angle2['pose'][:] = 0.
            # code_dict_angle2['pose'][:,1] = -np.pi/12
            # code_dict_angle2['cam'][:, 1:] = 0.
            # # get codedict with pose pi/6 and camera zeros
            # code_dict_angle3= copy.deepcopy(codedict)
            # code_dict_angle3['pose'][:] = 0.
            # code_dict_angle3['pose'][:,1] = np.pi/12
            # code_dict_angle3['cam'][:, 1:] = 0.
            # # get codedict with pose -pi/12 and camera zeros
            # code_dict_angle4= copy.deepcopy(codedict)
            # code_dict_angle4['pose'][:] = 0.
            # code_dict_angle4['pose'][:,1] = -np.pi/12
            # code_dict_angle4['cam'][:, 1:] = 0.
            #             # get codedict with pose pi/12 and camera zeros
            # code_dict_angle5= copy.deepcopy(codedict)
            # code_dict_angle5['pose'][:] = 0.
            # code_dict_angle5['pose'][:,1] = np.pi/12
            # code_dict_angle5['cam'][:, 1:] = 0.
            # # get codedict with pose -pi/12 and camera zeros
            # code_dict_angle6= copy.deepcopy(codedict)
            # code_dict_angle6['pose'][:] = 0.
            # code_dict_angle6['pose'][:,1] = -np.pi/12
            # code_dict_angle6['cam'][:, 1:] = 0.

            # decode albedo
            tex_3dmm = model.decode_albedo(codedict)

            # get flame decode output dictionary for 3dmm texture
            opdict_3dmm, visdict_3dmm = model.decode(codedict, rendering = True, vis_lmk=False, return_vis=True)

            # get flame decode output dictionary for 3dmm texture at pose zeros, camera zeros
            opdict_3dmm_align, visdict_3dmm_align = model.decode(code_dict_align, rendering = True, vis_lmk=False, return_vis=True)
        
            # get texture from image
            texture_imgs, texture_vis_masks = util.compute_texture_map(org_image, opdict_3dmm['verts'], opdict_3dmm['normals'], 
                                                                     codedict['cam'], texture_data)
            
            tex_face = texture_imgs * texture_vis_masks

            tex_fill_flip = fill_flip(tex_face)

            input_fine_tex_model = torch.cat([tex_3dmm, tex_fill_flip], dim=1)

            fine_texture = skin_model(input_fine_tex_model)

            fine_texture_lighting = lighting_uv_tex(fine_texture, visdict_3dmm['uv_detail_normals'], codedict['lights'])
            
            tex_3dmm_lighting = lighting_uv_tex(tex_3dmm, visdict_3dmm['normal_uv'], codedict['lights'])

            tex_3dmm_lighting_fine_normal = lighting_uv_tex(tex_3dmm, visdict_3dmm['uv_detail_normals'], codedict['lights'])
            tex_3dmm_lighting_fine_normal = lighting_uv_tex(fine_texture, visdict_3dmm['uv_detail_normals'], codedict['lights'])

            # decode with fine texture have lighting
            opdict_fine_texture, visdict_fine_texture = model.decode(codedict, fine_tex=fine_texture_lighting)

            # decoder with fine texture have lighting at pose zeros, camera zeros
            opdict_fine_texture_align, visdict_fine_texture_align = model.decode(code_dict_align, fine_tex=fine_texture_lighting)

            # decode with pixel from image
            opdict_face_pixel, visdict_face_pixel = model.decode(codedict, fine_tex=tex_face)

            # decode with pixel from image at pose zeros, camera zeros
            opdict_face_pixel_align, visdict_face_pixel_align = model.decode(code_dict_align, fine_tex=tex_face)

        #-- save results for evaluation
        verts = opdict_fine_texture['verts'].cpu().numpy()
        landmark_51 = opdict_fine_texture['landmarks3d_world'][:, 17:]
        landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
        landmark_7 = landmark_7.cpu().numpy()


        visdict = {}
        visdict["inputs"] = visdict_fine_texture['inputs']
        visdict["landmarks3d"] = visdict_fine_texture["landmarks3d"]
        visdict["3dmm"] = visdict_3dmm['rendered_images']
        visdict["3dmm_align"] = visdict_3dmm_align['rendered_images']
        visdict["face_pixel"] = visdict_face_pixel['rendered_images']
        visdict["face_pixel_align"] = visdict_face_pixel_align['rendered_images']
        visdict["fine_texture"] = visdict_fine_texture['rendered_images']
        visdict["fine_texture_align"] = visdict_fine_texture_align['rendered_images']


        for k in range(images.shape[0]):
            os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
            # save codedict
            np.save(os.path.join(savefolder, f'{imagename[k]}_codedict.npy'), infer_code[k].detach().cpu().numpy())
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

             # compute loss
            #distance_3dmm = id_loss(visdict_3dmm_align['rendered_images'][k:k+1], visdict['inputs'][k:k+1]) 

            # get regcognition feature
            target = id_loss.reg_features(id_loss.transform(visdict['inputs'][k:k+1]))
            _3dmm = id_loss.reg_features(id_loss.transform(visdict["3dmm"][k:k+1]))
            _3dmm_align = id_loss.reg_features(id_loss.transform(visdict["3dmm_align"][k:k+1]))
            face_pixel = id_loss.reg_features(id_loss.transform(visdict["face_pixel"][k:k+1]))
            face_pixel_align = id_loss.reg_features(id_loss.transform(visdict["face_pixel_align"][k:k+1]))
            fine_tex = id_loss.reg_features(id_loss.transform(visdict["fine_texture"][k:k+1]))
            fine_tex_align = id_loss.reg_features(id_loss.transform(visdict["fine_texture_align"][k:k+1]))

            distance_3dmm = id_loss.cos_metric_evaluate(_3dmm, target)
            distance_face_pixel = id_loss.cos_metric_evaluate(face_pixel, target)
            distance_fine_tex = id_loss.cos_metric_evaluate(fine_tex, target)

            distance_3dmm_align = id_loss.cos_metric_evaluate(_3dmm_align, target)
            distance_face_pixel_align = id_loss.cos_metric_evaluate(face_pixel_align, target)
            distance_fine_tex_align = id_loss.cos_metric_evaluate(fine_tex_align, target)

            #logger.info(f"{imagename[k]} 3dmm: {distance_3dmm}\t fine tex: {distance_fine}\t tex_face: {distance_tex_from_face}\n")    
            
            # Create a dictionary of numpy to store regcognition feature
            data = {
                'target': target.detach().cpu().numpy()[0],
                '_3dmm': _3dmm.detach().cpu().numpy()[0],
                '_3dmm_align': _3dmm_align.detach().cpu().numpy()[0],
                'face_pixel': face_pixel.detach().cpu().numpy()[0],
                'face_pixel_align': face_pixel_align.detach().cpu().numpy()[0],
                'fine_tex': fine_tex.detach().cpu().numpy()[0],
                'fine_tex_align': fine_tex_align.detach().cpu().numpy()[0],
            }

            distance = {
                'image_name' : imagename[k],
                'distance_3dmm': distance_3dmm.detach().cpu().numpy()[0],
                'distance_face_pixel': distance_face_pixel.detach().cpu().numpy()[0],
                'distance_fine_tex': distance_fine_tex.detach().cpu().numpy()[0],
                'distance_3dmm_align': distance_3dmm_align.detach().cpu().numpy()[0],
                'distance_face_pixel_align': distance_face_pixel_align.detach().cpu().numpy()[0],
                'distance_fine_tex_align': distance_fine_tex_align.detach().cpu().numpy()[0],
            }
            
            print(distance_3dmm.shape)

            #logger.info(f"{imagename[k]} 3dmm: {distance_3dmm} face pixel: {distance_face_pixel} fine tex: {distance_fine_tex}\t") 

            distance_infor = ""
            for key, v in distance.items():
                if key == 'image_name':
                    distance_infor += f'{key}: v'
                else:
                    distance_infor = distance_infor + f'{key}: {v:.6f}, '                   
            logger.info(distance_infor)

            # Save the dictionary to a .npz file
            save_path = os.path.join(savefolder, f'{imagename[k]}_output_regcognition_our.npz')
            np.savez(save_path, **data)

            writer.writerow(distance.values())

        util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))

if __name__ == "__main__":
    logger.add(os.path.join(cfg.log_dir, 'tex_distance.log'))
    id_loss = loss.VGGFace2Loss(pretrained_model=cfg.Cself.fr_model_path)   
    # Open the CSV file in write mode
    csvfile = open('output.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    evaluate(id_loss, writer)
    # Close the CSV file
    csvfile.close()
    print("CSV file saved successfully.")