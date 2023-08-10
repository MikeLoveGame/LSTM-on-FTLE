# Import numpy
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def save_3Darray_as_raw(array, destination_file):
    # Flatten the array to a 1-dimensional array
    flattened_array = []
    for i in range(array.particle_shape[0]):
        for j in range(array.particle_shape[1]):
            for k in range(array.particle_shape[2]):
                flattened_array.append(float(array[i, j, k]))
    flattened_array=np.asarray(flattened_array, dtype=np.float32)
    # Convert the array elements to bytes
    bytes_array = flattened_array.tobytes()

    # Save the bytes to the destination file
    with open(destination_file, 'wb') as file:
        file.write(bytes_array)

source=r"C:\AI\CNIC\SAM\Data\temp"
destination=r"C:\AI\CNIC\SAM\Data\tempData"
img_files = os.listdir(source)
data=np.zeros(shape=(30, 512, 512), dtype=np.float32)
z=0;
for img in img_files:
    print(img)
    img_path = os.path.join(source, img)
    img_data=cv2.imread(img_path)
    img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_2D = np.sum(img_data[:, :], axis=2)
    data[z]=img_2D
    z=z+1
data = data.transpose(range(data.ndim)[::-1])
data = torch.from_numpy(data)
translated_data = F.interpolate(input=data, mode="linear", scale_factor=8)

save_3Darray_as_raw(translated_data, os.path.join(destination,"chest_CT"+".raw"))