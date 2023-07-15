import csv
import os

def copy_invalid_files(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        rows_to_copy = []

        for row in reader:
            img_path_name = row[0]
            normal_path = img_path_name.replace("_visualize.jpg", "_normal_uv.npy")
            if not os.path.exists(normal_path):
                rows_to_copy.append(row)

    if rows_to_copy:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows_to_copy)
            print(f"Invalid file paths copied to '{output_file}'.")
    else:
        print("All file paths are valid.")

# Usage
input_csv_file = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/train.csv'  # Replace with the path to your input CSV file
output_csv_file = '/root/baonguyen/3d_face_reconstruction/datasets/BUPT/FOCUS/train_remain.csv'  # Replace with the path to the output CSV file

copy_invalid_files(input_csv_file, output_csv_file)