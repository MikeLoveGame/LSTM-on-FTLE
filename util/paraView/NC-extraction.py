import netCDF4 as nc
import numpy as np
import os

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


def save_as_numpy(array: list, destination_file : str):
    nparr = np.asarray(a=array, dtype=np.float32)
    np.save(file=destination_file, arr=nparr, allow_pickle=True)

f=nc.Dataset(r"C:\AI\Data\Field-data\0001\0001\COMBINED_2011013100.nc")
vec_U=f.variables["U"]
vec_V=f.variables["V"]
V_filename="V-vec"
U_filename="U-vec"
dir_destination=r"C:\AI\Data\Field-data\UV-vector-numpy"


for i in range(vec_V.particle_shape[0]):
    path=os.path.join(dir_destination, V_filename+str(i)+".npy")
    save_as_numpy(array=vec_V[i], destination_file=path)
    print(path + str(i))

for i in range(vec_U.particle_shape[0]):
    path=os.path.join(dir_destination, U_filename+str(i)+".npy")
    save_as_numpy(array=vec_U[i], destination_file=path)
    print(path + str(i))








