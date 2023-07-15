import os
import csv
import random
from tqdm import tqdm

def check_valid_files(input_file, output_file):
    # Open the CSV files for writing
    outfile = open(output_file, "w", newline="")

    # Create CSV writers
    train_writer = csv.writer(outfile)

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            img_path_name = row[0]

            coarse_code_path = img_path_name.replace("_visualize.jpg", "_coarse_code.npz")
            tex_skin_seg_path = img_path_name.replace("_visualize.jpg", "_tex_skin_seg.npz")
            normal_uv_path = img_path_name.replace("_visualize.jpg", "_normal_uv.npz")
            tex_3dmm_path = img_path_name.replace("_visualize.jpg", "_tex_3dmm.jpg")
            tex_face_path =  img_path_name.replace("_visualize.jpg", "_tex_face.jpg")

            # Check if all files exist
            if os.path.exists(coarse_code_path) and os.path.exists(tex_skin_seg_path) and \
                os.path.exists(normal_uv_path) and os.path.exists(tex_3dmm_path) and os.path.exists(tex_face_path):
                train_writer.writerow([coarse_code_path, tex_3dmm_path, tex_face_path,
                                        tex_skin_seg_path, normal_uv_path])
    outfile.close()

# Usage
input_csv_file = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/val.csv'  # Replace with the path to your input CSV file
output_csv_file = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/val_tex.csv'  # Replace with the path to the output CSV file

check_valid_files(input_csv_file, output_csv_file)

