
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

import scipy
from glob import glob
from skimage.io import imread, imsave
from torch.utils.data import Dataset
from skimage.transform import estimate_transform, warp, resize, rescale
from torchvision import transforms
from PIL import Image
import torchvision
import face_alignment

from datasets import detectors

from utils import util
from utils.config import cfg
from coarse_model import C_model
from datasets.coarse_infer_loader import InferDataset

from utils.renderer import SRenderY, set_rasterizer
import torch.nn.functional as F
from datasets.test_data import TestData
import cv2
from models.Texture_fine_net import Fine_net


def bbox2point(left, right, top, bottom, type='bbox'):
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

def load_mask(maskpath, h, w):
    # print(maskpath)
    if os.path.isfile(maskpath):
        vis_parsing_anno = np.load(maskpath)['arr_0']
        mask = np.zeros_like(vis_parsing_anno)
        # for i in range(1, 16):
        mask[vis_parsing_anno>0.5] = 1.
    else:
        mask = np.ones((h, w))
    return mask

def detect_kpt_for_crop_img(imagepath):
    image = np.array(imread(imagepath))
    if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:,:,:3]
    kpts = fan.get_landmarks(image)
    if kpts == None:
        return
    # Save the keypoints as a numpy array
    np.savetxt(os.path.splitext(imagepath)[0]+'.txt', kpts[0]) 
        
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

def process_mask(np_mask, threshold):
    mask = np.zeros_like(np_mask)
    # for i in range(1, 16):
    mask[np_mask>threshold] = 1.
    mask = mask[:,:,0:1]
    return mask    

def getitem(path_input):
    imagepath = path_input
    imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
    image = np.array(imread(imagepath))
    img_PIL = Image.open(imagepath).convert('RGB')
    if len(image.shape) == 2:
        image = image[:,:,None].repeat(1,1,3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:,:,:3]

    h, w, _ = image.shape

    scale=1.25
    resolution_inp=224
    transform_input = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # if cfg.Test.use_skin_seg:
    #     # Load segmentation
    #     p_tex_skin_seg = os.path.splitext(imagepath)[0]+'.npz'
    #     tex_skin_seg = load_mask(p_tex_skin_seg, h, w)

    # provide kpt as txt file, or mat file (for AFLW2000)
    kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
    kpt = np.loadtxt(kpt_txtpath)
    left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
    old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
    
    size = int(old_size*scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

    DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    image = image/255.

    dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    dst_image = dst_image.transpose(2,0,1)
    
    tensor_img = transform_input(torch.tensor(dst_image).float())

    output = {'image': torch.unsqueeze(tensor_img,0),
            'imagename': imagename,
            'tform': torch.unsqueeze(torch.tensor(tform.params).float(),0),
            'original_image': torch.unsqueeze(torch.tensor(dst_image).float(),0),
            }
    return output

# input is a image file path
def infer(path_input):

    print('loading test image')
    testdata = getitem(path_input)
    output_dir = os.path.dirname(path_input)
    os.makedirs(os.path.join(output_dir, 'infer'), exist_ok=True)
    savefolder = os.path.join(output_dir, 'infer', f'') 
    os.makedirs(savefolder, exist_ok=True)

    image = testdata['image'].to(device)
    org_image = testdata['original_image'].to(device)
    imagename = testdata['imagename']

    # get texture data
    texture_data = np.load(cfg.flame.dense_template_path, allow_pickle=True, encoding='latin1').item()

    print('infering')
    with torch.no_grad():
        # encode image to coarse code
        print('encode image to coarse code')
        codedict, parameters = model.encode(image)
        codedict['images'] = org_image

        # decode albedo
        tex_3dmm = model.decode_albedo(codedict)

        # get flame decode output dictionary
        print('get flame decode output dictionary')
        opdict, visdict_0 = model.decode(codedict, rendering = True, vis_lmk=False, return_vis=True)

        texture_imgs, texture_vis_masks = util.compute_texture_map(org_image, opdict['verts'], opdict['normals'], 
                                                                    codedict['cam'], texture_data)
        tex_face = texture_imgs * texture_vis_masks

        input_fine_tex_model = torch.cat([tex_3dmm, tex_face], dim=1)
        print('fine_tex infering')
        fine_texture = skin_model(input_fine_tex_model)

        fine_texture_lighting = lighting_uv_tex(fine_texture, visdict_0['uv_detail_normals'], codedict['lights'])

        opdict, visdict = model.decode(codedict, fine_tex=fine_texture_lighting)


        #-- save coarse shape results
    verts = opdict['verts'].cpu().numpy()
    
    print('saving')
    os.makedirs(os.path.join(savefolder, imagename), exist_ok=True)
    # save mesh
    util.write_obj(os.path.join(savefolder, imagename, f'{imagename}.obj'), vertices=verts[0], faces=faces)
    
    #-- save fine shape result
    dense_vertices = opdict['dense_vertices'].cpu().numpy()
    fine_texture_np = fine_texture_lighting.permute(0,2, 3, 1).detach().cpu().numpy()*255
    
    path_output = os.path.join(savefolder, imagename, f'{imagename}_fine.obj')

    tex_3dmm_path = os.path.join(savefolder, imagename, f'{imagename}_3dmm_uv.jpg')
    tex_org_path = os.path.join(savefolder, imagename, f'{imagename}_origin_uv.jpg')

    # save 3DMM UV lighting
    tex_3dmm_lighting = lighting_uv_tex(tex_3dmm, visdict_0['uv_detail_normals'], codedict['lights'])
    tex_3dmm_np = tex_3dmm_lighting.permute(0,2, 3, 1).detach().cpu().numpy()*255
    cv2.imwrite(tex_3dmm_path, tex_3dmm_np[0][:, :, ::-1])
    # save vis_org_texture 
    tex_org_np = texture_imgs.permute(0,2, 3, 1).detach().cpu().numpy()*255
    cv2.imwrite(tex_org_path, tex_org_np[0][:, :, ::-1])

    # save dense_mesh
    util.write_obj(path_output,
                    vertices=dense_vertices[0],
                    faces=dense_faces,
                    colors=None,
                    texture=fine_texture_np[0][:, :, ::-1],
                    uvcoords=uv_dense_vertices,
                    uvfaces=dense_faces,
                    inverse_face_order=False,
                    normal_map=None,
                    )
    # return path to file .obj, file uv_3dmm, file uv_original image
    return path_output, tex_3dmm_path, tex_org_path

def printCudaState():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Total CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"Used CUDA Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Free CUDA Memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    else:
        print("CUDA is not available on this system.")


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/Reconstruct3dFace', methods=['POST'])
def Reconstruct3dFace():
    data = request.json  # Get the JSON data from the request

    # Process the data or perform any desired actions
    file_path = data['input'] 
    for i in tqdm(range(1)):
        kpt_txtpath = os.path.splitext(file_path)[0]+'.txt'
        if not os.path.exists(kpt_txtpath):
            print('detecting keypoints for crop image')
            detect_kpt_for_crop_img(file_path)
        print('start infering')
        obj_path, uv_3dmm_path, uv_org_path = infer(file_path)

    # Return the result as a JSON response
    response = {'obj_path': obj_path,
                'uv_3dmm_path': uv_3dmm_path,
                "uv_org_path": uv_org_path,}
    return jsonify(response)

def test():
    
    # Process the data or perform any desired actions
    file_path = "D:/University/LUAN_VAN/code/our3DFaceReconstruct/input/non_skin_seg/justin.png"

    kpt_txtpath = os.path.splitext(file_path)[0]+'.txt'
    if not os.path.exists(kpt_txtpath):
        print('detecting keypoints for crop image')
        for i in tqdm(range(1),desc='detect'):
            detect_kpt_for_crop_img(file_path)

    print('start infering')
    for i in tqdm(range(1),desc='infer'):
        result, rl1, rl2 = infer(file_path)
    print(result)

def save_state_dict(model_path, state_dict_path):
    if os.path.exists(model_path):
        print(f'trained model found. load {model_path}')
        checkpoint = torch.load(model_path)
        torch.save(checkpoint['state_dict'],state_dict_path)

def initialize_model():
    # run C_model
    global device
    global model, skin_model, fan
    global faces, dense_faces, uv_dense_vertices, texture_data

    # initialize Face detect landmark models: FAN
    fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    device = cfg.device
    # run skin_fine model
    skin_model = Fine_net().to(device)
    # initialize fine texture reconstruct model
    checkpoint = torch.load(cfg.finetex.state_dict_path)
    skin_model.load_state_dict(checkpoint)
    # initialize shape reconstruct model
    model = C_model(config = cfg, device=device, use_fine_shape=True)
    model.eval()


    # coarse faces mesh
    faces = model.flame.faces_tensor.cpu().numpy()
    # Load details faces mesh infor
    dense_infor = np.load(cfg.flame.dense_infor_path)

    dense_faces = dense_infor['dense_face']

    uv_dense_vertices = dense_infor['uv_coords']

if __name__ == '__main__':
    initialize_model()
    # run API to communicate to Blender plugin
    app.run()


    