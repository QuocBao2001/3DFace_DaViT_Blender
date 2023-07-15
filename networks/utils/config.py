'''
Default config
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import torch

cfg = CN()
cfg.exp_name = 'change input increase eyes loss'
cfg.working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.flame_resource = os.path.join(cfg.working_dir, 'data', 'flame_resource')

cfg.count_time = 0

#cfg.output_dir = os.path.join(cfg.working_dir, 'output', '23_06_change_input')
cfg.output_dir = os.path.join(cfg.working_dir, 'output', '04_07_increase_eyes_loss')
#cfg.output_dir = os.path.join(cfg.working_dir, 'output', 'test_grid')
cfg.log_dir = os.path.join(cfg.output_dir,'logs')
cfg.coarse_model_dir = os.path.join(cfg.output_dir, 'C_models')
cfg.fine_model_dir = os.path.join(cfg.output_dir, 'F_models')
cfg.fine_shape_model_path = "/root/baonguyen/3d_face_reconstruction/data/DECA_pretrained/detail_pretrained_model.tar"
cfg.VGGFacePath = "/root/baonguyen/3d_face_reconstruction/data/VGGFace/VGG_FACE.t7"

# ---------------------------------------------------------------------------- #
# config for dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.image_size = 224
cfg.dataset.trans_scale = 0.

# ---------------------------------------------------------------------------- #
# config for flame
# ---------------------------------------------------------------------------- #
cfg.flame = CN()
cfg.flame.topology_path = os.path.join(cfg.flame_resource, 'head_template_mesh.obj')
cfg.flame.flame_model_path = os.path.join(cfg.flame_resource, 'generic_model.pkl')
cfg.flame.dense_template_path = os.path.join(cfg.flame_resource, 'texture_data_256.npy')
cfg.flame.fixed_displacement_path = os.path.join(cfg.flame_resource, 'fixed_displacement_256.npy')
cfg.flame.flame_lmk_embedding_path = os.path.join(cfg.flame_resource, 'landmark_embedding.npy') 
cfg.flame.face_eye_mask_path = os.path.join(cfg.flame_resource, 'uv_face_eye_mask.png') 
cfg.flame.face_wthout_eye = os.path.join(cfg.flame_resource, 'uv_wthout_eyes.jpg') 
cfg.flame.mask_right_eye_path = os.path.join(cfg.flame_resource, 'eye_right.jpg')
cfg.flame.mask_left_eye_path = os.path.join(cfg.flame_resource, 'eye_left.jpg')
cfg.flame.dense_infor_path = os.path.join(cfg.flame_resource, 'dense_infor.npz')
cfg.flame.use_tex = False #True
cfg.flame.tex_type = 'FLAME' 
cfg.flame.flame_tex_path = os.path.join(cfg.flame_resource, 'FLAME_texture.npz')
cfg.flame.uv_size = 256
cfg.flame.rasterizer_type = 'pytorch3d'
cfg.flame.param_list = ['shape', 'exp', 'pose', 'texture_code', 'light_code', 'cam']

cfg.flame.n_shape = 100
cfg.flame.n_exp = 50
cfg.flame.n_pose = 6

cfg.flame.n_tex = 50
cfg.flame.n_light = 27
cfg.flame.n_cam = 3

cfg.flame.total_params = cfg.flame.n_shape + cfg.flame.n_exp + cfg.flame.n_pose + \
                            + cfg.flame.n_tex + cfg.flame.n_light + cfg.flame.n_cam

cfg.flame.vis_dir = os.path.join(cfg.output_dir, 'visualize')

# ---------------------------------------------------------------------------- #
# config for fine texture model
# ---------------------------------------------------------------------------- #
cfg.finetex = CN()
cfg.finetex.input_channels = 6
cfg.finetex.img_size = 256
cfg.finetex.csv_val = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/val_tex.csv"
cfg.finetex.csv_train = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/train_tex.csv"
cfg.finetex.vis_dir = os.path.join(cfg.output_dir, 'visualize')
cfg.finetex.batch_size = 8
cfg.finetex.num_worker = 1
cfg.finetex.img_size = 256

cfg.finetex.lr = 5e-6
cfg.finetex.resume = True
cfg.finetex.checkpoint_name = 'best_at_00266500.tar'
cfg.finetex.checkpoint_path = os.path.join(cfg.coarse_model_dir, cfg.finetex.checkpoint_name)
# cfg.finetex.checkpoint_path = '/root/baonguyen/3d_face_reconstruction/output/23_06_increase_symmetric/F_models/best_at_00082500.tar'
# cfg.finetex.pretrained_path = '/root/baonguyen/3d_face_reconstruction/output/23_06_increase_symmetric/F_models/best_at_00082500.tar'

cfg.finetex.checkpoint_path = '/root/baonguyen/3d_face_reconstruction/output/23_06_change_input/F_models/best_at_00124000.tar'
cfg.finetex.pretrained_path = '/root/baonguyen/3d_face_reconstruction/output/23_06_change_input/F_models/best_at_00124000.tar'

# cfg.finetex.checkpoint_path = "/root/baonguyen/3d_face_reconstruction/output/15_06_tex_eye_loss_fix/F_models/00036000.tar"
# cfg.finetex.pretrained_path = '/root/baonguyen/3d_face_reconstruction/output/15_06_tex_eye_loss_fix/F_models/00036000.tar'

cfg.finetex.write_summary = True

cfg.finetex.num_epochs = 50
cfg.finetex.num_steps = 10000000

cfg.finetex.log_steps = 100
cfg.finetex.checkpoint_steps = 500
cfg.finetex.val_steps = 500
cfg.finetex.vis_steps = 100
cfg.finetex.plot_steps = 1000
cfg.finetex.train_record_steps = 200

cfg.finetex.eval_steps = 5000


cfg.finetex.pixel_per_eye = 926
cfg.finetex.ratio_occlution = 0.3
# loss function weight 
cfg.finetex.sym_loss = 3
cfg.finetex.face_loss = 2.
cfg.finetex.tex_3dmm_loss = 0.5
cfg.finetex.bound_loss = 1.
cfg.finetex.transfer_3dmm_loss = 0.5
cfg.finetex.eyes_loss = 20.
cfg.finetex.mrfloss = .02

cfg.finetex.bound_thickness = 9

# ---------------------------------------------------------------------------- #
# config for coarse model with supervised training
# ---------------------------------------------------------------------------- #
cfg.Csup = CN()
cfg.Csup.batch_size = 16
cfg.Csup.num_worker = 2
cfg.Csup.img_size = 224
cfg.Csup.lr = 5e-5
cfg.Csup.resume = True
cfg.Csup.checkpoint_name = 'best_at_00266500.tar'
cfg.Csup.checkpoint_path = os.path.join(cfg.coarse_model_dir, cfg.Csup.checkpoint_name)
cfg.Csup.checkpoint_path = "/root/baonguyen/3d_face_reconstruction/output/27_05_only_real/C_models/best_at_00308000.tar"
cfg.Csup.pretrained_path = os.path.join(cfg.working_dir, 'data', 'DaViT', 'DaViT_Encoder_236.pth.tar')
cfg.Csup.write_summary = True

cfg.Csup.loss = CN()
cfg.Csup.loss.shape_w = 3.
cfg.Csup.loss.exp_w = 2.
cfg.Csup.loss.pose_w = 1.

cfg.Csup.loss.tex_w = 3.
cfg.Csup.loss.light_w = 1.
cfg.Csup.loss.cam_w = 1.

cfg.Csup.dataset_dir = os.path.join(cfg.working_dir, 'datasets', 'coarse_supervised')
cfg.Csup.train_dir = os.path.join(cfg.Csup.dataset_dir, 'train')
cfg.Csup.val_dir = os.path.join(cfg.Csup.dataset_dir, 'val')

cfg.Csup.num_epochs = 50
cfg.Csup.num_steps = 10000000

cfg.Csup.log_steps = 100
cfg.Csup.checkpoint_steps = 500
cfg.Csup.val_steps = 500
cfg.Csup.vis_steps = 1000
cfg.Csup.plot_steps = 1000
cfg.Csup.train_record_steps = 200

cfg.Csup.eval_steps = 5000

# ---------------------------------------------------------------------------- #
# config for coarse model with self-supervised training
# ---------------------------------------------------------------------------- #
cfg.Cself = CN()
cfg.Cself.batch_size = 16
cfg.Cself.num_worker = 2
cfg.Cself.img_size = 224
cfg.Cself.lr = 5e-6
cfg.Cself.resume = True
cfg.Cself.checkpoint_name = 'best_at_00266500.tar'
cfg.Cself.checkpoint_path = os.path.join(cfg.coarse_model_dir, cfg.Csup.checkpoint_name)
cfg.Cself.checkpoint_path = "/root/baonguyen/3d_face_reconstruction/output/02_06_self_sup_fix/C_models/best_at_00069000.tar"
cfg.Cself.pretrained_path = os.path.join(cfg.working_dir, 'data', 'DaViT', 'DaViT_Encoder_236.pth.tar')
cfg.Cself.pretrained_path = "/root/baonguyen/3d_face_reconstruction/output/02_06_self_sup_fix/C_models/best_at_00069000.tar"
cfg.Cself.csv_train_path = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/train.csv"
cfg.Cself.csv_val_path = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/val.csv"

cfg.Cself.loss = CN()
# loss for self supervise learning
cfg.Cself.loss.useWlmk = True
cfg.Cself.loss.lmk = 1.0
cfg.Cself.loss.photo = 2.0
cfg.Cself.loss.useSeg = True
cfg.Cself.loss.id = 0.2
cfg.Cself.fr_model_path = '/root/baonguyen/3d_face_reconstruction/data/id_feature_extract/resnet50_ft_weight.pkl'

# loss for regularization output
cfg.Cself.loss.reg_shape = 1e-04
cfg.Cself.loss.reg_exp = 1e-04
cfg.Cself.loss.reg_tex = 1e-04
cfg.Cself.loss.reg_light = 1.
cfg.Cself.loss.reg_jaw_pose = 0. #1.

cfg.Cself.num_epochs = 50
cfg.Cself.num_steps = 10000000

cfg.Cself.log_steps = 100
cfg.Cself.checkpoint_steps = 500
cfg.Cself.val_steps = 500
cfg.Cself.vis_steps = 1000
cfg.Cself.plot_steps = 1000
cfg.Cself.train_record_steps = 200

cfg.Cself.eval_steps = 5000

# ---------------------------------------------------------------------------- #
# config for fine model with supervised training
# ---------------------------------------------------------------------------- #
cfg.Fsub = CN()

# ---------------------------------------------------------------------------- #
# config for test model
# ---------------------------------------------------------------------------- #
cfg.Test = CN()
cfg.Test.savefolder = os.path.join(cfg.output_dir, 'test')
cfg.Test.input_paths = '/root/baonguyen/3d_face_reconstruction/datasets/test_img'
cfg.Test.iscrop = True
cfg.Test_detect_kpt = True
cfg.Test.detector = 'fan'
cfg.Test.sample_step = 10


# ---------------------------------------------------------------------------- #
# config for infering, API, please fill following path
# ---------------------------------------------------------------------------- #

#cfg.fine_shape_model_path =
#cfg.finetex.checkpoint_path =
#cfg.finetex.state_dict_path = 
#cfg.Cself.checkpoint_path =
#cfg.Cself.state_dict_path = 

cfg.isInfer = False
# ---------------------------------------------------------------------------- #


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
