import csv
import os
from tqdm import tqdm

csv_file_path = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/train_tex.csv"

# Specify the column index (zero-based) that contains the image paths
image_path_column_index = 0

# Read the CSV file
with open(csv_file_path, "r") as file:
    reader = csv.reader(file)
    
    
    # Iterate over each row in the CSV file
    for row in tqdm(reader):
        if len(row) > image_path_column_index:
            image_path = row[image_path_column_index]
            
            # Check if the image path is valid
            if os.path.isfile(image_path) and image_path.endswith('_coarse_code.npy'):
                # Delete the image file
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            else:
                print(f"File not found: {image_path}")