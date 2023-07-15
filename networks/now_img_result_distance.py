import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import loss
from utils.config import cfg
import csv

def get_image_pairs(root_folder):
    image_pairs = []
    
    for person_folder in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        for pose_folder in os.listdir(person_path):
            pose_path = os.path.join(person_path, pose_folder)
            if not os.path.isdir(pose_path):
                continue

            for img_id_folder in os.listdir(pose_path):
                id_path = os.path.join(pose_path, img_id_folder)
                if not os.path.isdir(id_path):
                    continue
            
                image_files = os.listdir(id_path)
                for image_file in image_files:
                    image_path = os.path.join(id_path, image_file)
                    if image_file.endswith('_inputs.jpg'):
                        origin_path = image_path
                    elif image_file.endswith('_rendered_images.jpg'):
                        reconstruction_path = image_path
                
                if not (os.path.isfile(origin_path) and os.path.isfile(reconstruction_path)):
                    continue
                
                print(origin_path)
                print(reconstruction_path)
                image_pairs.append((origin_path, reconstruction_path))
    
    return image_pairs


root_folder = '/root/baonguyen/3d_face_reconstruction/output/FOCUS_result'
image_pairs = get_image_pairs(root_folder)

id_loss = loss.VGGFace2Loss(pretrained_model=cfg.Cself.fr_model_path)   

transform = transforms.Compose([
    transforms.ToTensor()
])

# Create a CSV file to save the results
csv_file = open('result_FOCUS.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

for image_path1, image_path2 in image_pairs:
    # Load the first image using PIL
    image1 = Image.open(image_path1)
    # Apply the transformation to the first image
    tensor1 = transform(image1)

    # Load the second image using PIL
    image2 = Image.open(image_path2)
    # Apply the transformation to the second image
    tensor2 = transform(image2)

    # Move the tensors to CUDA if available
    if torch.cuda.is_available():
        tensor1 = tensor1.unsqueeze(0).cuda()
        tensor2 = tensor2.unsqueeze(0).cuda()

    # get distance
    distance = id_loss(tensor2, tensor1)

    distance = distance.detach().cpu().numpy()
    print(distance)
    print(image_path1)
    print(image_path2)

    # Write the result to the CSV file
    csv_writer.writerow([image_path1, distance])

# Close the CSV file
csv_file.close()