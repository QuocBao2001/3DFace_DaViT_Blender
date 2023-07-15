import os
import csv
import random
from tqdm import tqdm

# Set the path to the root folder
root_folder = "/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS"

# Define the output CSV filenames
train_csv_file = os.path.join(root_folder, "train.csv")
val_csv_file = os.path.join(root_folder, "val.csv")

# Open the CSV files for writing
train_csv = open(train_csv_file, "w", newline="")
val_csv = open(val_csv_file, "w", newline="")

# Create CSV writers
train_writer = csv.writer(train_csv)
val_writer = csv.writer(val_csv)

# Write the header row for CSV files
# header = ["image_path", "lmks_path", "mask_path"]
# train_writer.writerow(header)
# val_writer.writerow(header)

# Loop through the root folder
for race_folder in os.listdir(root_folder):
    race_folder_path = os.path.join(root_folder, race_folder)
    if not os.path.isdir(race_folder_path):
        continue

    # Create a dictionary to store images per person
    person_images = {}

    # Loop through each person folder
    for person_folder in os.listdir(race_folder_path):
        person_folder_path = os.path.join(race_folder_path, person_folder)
        if not os.path.isdir(person_folder_path):
            continue
        
        data_folder = os.path.join(person_folder_path, 'seg_results')

        # Get the list of image files in the person folder
        image_files = [f for f in os.listdir(data_folder) if f.endswith("_visualize.jpg")]

        # Store the image files in the person_images dictionary
        person_images[person_folder] = image_files

    # Shuffle the list of person folders
    person_folders = list(person_images.keys())
    random.shuffle(person_folders)

    # split training and validation
    num_val = 100
    train_persons = person_folders[num_val:]
    val_persons = person_folders[:num_val]

    # Write the image paths, lmks paths, and mask paths to the respective CSV files
    for person_folder in tqdm(train_persons):
        image_files = person_images[person_folder]
        for image_file in image_files:
            image_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file)
            lmks_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file.replace("_visualize.jpg", "_lmks.npy"))
            mask_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file.replace("_visualize.jpg", "_results.npy"))

            # Check if lmks and mask files exist
            if os.path.exists(lmks_path) and os.path.exists(mask_path):
                train_writer.writerow([image_path, lmks_path, mask_path])

    for person_folder in tqdm(val_persons):
        image_files = person_images[person_folder]
        for image_file in image_files:
            image_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file)
            lmks_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file.replace("_visualize.jpg", "_lmks.npy"))
            mask_path = os.path.join(race_folder_path, person_folder, 'seg_results', image_file.replace("_visualize.jpg", "_results.npy"))

            # Check if lmks and mask files exist
            if os.path.exists(lmks_path) and os.path.exists(mask_path):
                val_writer.writerow([image_path, lmks_path, mask_path])

# Close the CSV files
train_csv.close()
val_csv.close()

print("CSV files created successfully!")
