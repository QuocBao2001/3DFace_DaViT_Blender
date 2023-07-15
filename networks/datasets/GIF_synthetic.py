import os 
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class GIFDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        if transform == None:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        else:
            self.transform = transform
        
        self.to_Tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.img_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'params')

        # get list of labels file name
        self.labels_list = [filename for filename in os.listdir(self.labels_dir) if filename.endswith('.npy')]
        self.labels_list = sorted(self.labels_list, key=lambda x: int(os.path.splitext(x)[0]))

        # store each numpy set label and image
        self.labels = []
        self.image_filenames = []
        # loop through all image and labels in each set
        for label_filename in self.labels_list:
            # get label set infor
            label_file = os.path.join(self.labels_dir, label_filename)
            current_label_set = self.npy2labels(label_file)
            
            # get the list of image filenames in set
            image_set_dir = os.path.join(self.img_dir, label_filename[:-4])
            curr_set_img_name = [filename for filename in os.listdir(image_set_dir) if filename.endswith('.png')]
            curr_set_img_name = sorted(curr_set_img_name, key=lambda x: int(os.path.splitext(x)[0]))
            curr_set_img_name = [os.path.join(label_filename[:-4],filename) for filename in curr_set_img_name]

            self.labels.append(current_label_set)
            self.image_filenames.extend(curr_set_img_name)
        
        # convert labels from list of np array to np array
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Read the image
        image_path = os.path.join(self.img_dir, self.image_filenames[idx])
        origin_image = Image.open(image_path).convert('RGB')

        # Get the corresponding label
        label = self.labels[idx]

        # Apply transformations to the image
        if self.transform:
            image = self.transform(origin_image)

        origin_image = self.to_Tensor(origin_image)

        return image, label, origin_image
    
    def npy2labels(self, file_name):
        # get dictionary's labels
        dict_labels = np.load(file_name, allow_pickle=True).item()
                
        # reshape light params
        dict_labels['light_code'] = dict_labels['light_code'].reshape((-1,27))
        
        # dict keys in order: ['shape', 'exp', 'pose', 'texture_code', 'light_code', 'cam'])
        labels = np.concatenate([dict_labels['shape'], dict_labels['exp'], 
                                      dict_labels['pose'], dict_labels['texture_code'], 
                                      dict_labels['light_code'], dict_labels['cam']], axis=1)
        
        return labels