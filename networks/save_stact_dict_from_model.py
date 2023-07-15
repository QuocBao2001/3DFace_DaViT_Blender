import torch
from tqdm import tqdm
import os

def save_state_dict(model_path):
    if os.path.exists(model_path):
        state_dict_path = os.path.splitext(model_path)[0] + 'state_dict.tar'
        print(f'model found. load {model_path}')
        checkpoint = torch.load(model_path)
        print(f'saing state_dict at {state_dict_path}')
        for i in tqdm(range(1),desc='saving state_dict of model'):
            torch.save(checkpoint['state_dict'],state_dict_path)

if __name__=='__main__':
    model_path = '/root/baonguyen/3d_face_reconstruction/output/23_06_change_input/F_models/best_at_00124000.tar'
    save_state_dict(model_path)