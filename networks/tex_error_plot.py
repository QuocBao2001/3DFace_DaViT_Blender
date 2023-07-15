import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_set(csv_file_name, load_indexs, subtract=False):
    distance_data = []
    with open(csv_file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if subtract:
                distances = [1 - float(distance) for distance in row[1:]]
            else:
                distances = [float(distance) for distance in row[1:]]
            distance_data.append(distances)

    np_array = np.array(distance_data)
    print(np_array.shape)
    mean = np.mean(np_array, axis=0)
    print(mean)

    # Sort the distance data in ascending order for each error type
    sorted_distance_data = [sorted(distances) for distances in zip(*distance_data)]

    # Calculate the percentage of images with distance less than or equal to x for each error type
    total_images = len(distance_data)
    percent_distances = [[(i + 1) / total_images * 100 for i in range(total_images)] for _ in range(len(distance_data[0]))]

    # Create the diagram
    for i, (distances, percent_distances) in enumerate(zip(sorted_distance_data, percent_distances)):
        if i in load_indexs:
            plt.plot(distances, percent_distances)

# Read the CSV file
filename = '/root/baonguyen/3d_face_reconstruction/networks/output.csv'
filename_DECA = "/root/baonguyen/3d_face_reconstruction/networks/result_DECA.csv"
filename_FOCUS = "/root/baonguyen/3d_face_reconstruction/networks/result_FOCUS.csv"

plot_set(filename, [0, 1, 2], subtract=True)
plot_set(filename_DECA, [0])
plot_set(filename_FOCUS, [0])

# legend to compare with zeros
# plt.legend(['origin image', '3dmm', 'uv mapping', 'fine_texture',
#                 '3dmm pose 0', 'uv mapping pose 0', 'fine_texture pose 0'])

# legend to compare with other
# plt.legend(['origin image', #'3dmm', 'uv mapping', 'fine_texture',
#                 '3dmm pose 0', 'uv mapping pose 0', 'fine_texture pose 0'])

plt.legend(['Tái tạo thô', 'chiếu ảnh vào UV', 'Tái tạo tinh', 'DECA', 'FOCUS'])

# Set labels and title
plt.xlabel('Cosine distance')
plt.ylabel('Percent')
# Save the plot as an image file
plt.savefig('distance_diagram_DECA.png')

# Display the plot
plt.show()