import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def view_image(data: np.ndarray):
    plt.imshow(data)
    plt.show()

def save_3Darray_as_raw(array, destination_file):
    # Flatten the array to a 1-dimensional array
    flattened_array = []
    print(array.particle_shape)
    print(array.dtype)
    for i in range(array.particle_shape[0]):
        for j in range(array.particle_shape[1]):
            for k in range(array.particle_shape[2]):
                flattened_array.append(array[i, j, k])
    flattened_array=np.asarray(flattened_array, dtype=np.float32)
    # Convert the array elements to bytes
    bytes_array = flattened_array.tobytes()

    # Save the bytes to the destination file
    with open(destination_file, 'wb') as file:
        file.write(bytes_array)

source= r"D:\AI-Data\Field-Data\vector-field\0001\result0001\upper"

#time 33 oweddy data incompatiable

dirs=os.listdir(source)
for dir in dirs:
    path=os.path.join(source, dir)
    files = os.listdir(path)
    print(dir)
    for file in files:
        if (".npy" in file):
            data=np.load(os.path.join(path, file), allow_pickle=True)
            filename=file.strip(".npy")
            destination=os.path.join(path, filename+".raw")
            #save_3Darray_as_raw(array=data, destination_file=destination)