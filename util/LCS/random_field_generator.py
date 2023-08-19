import os
import numpy as np



def divergent_circle(location : (int, int), radius : int, vector : np.ndarray):
    dim = vector.shape
    if (len(dim) != 2):
        raise Exception("vector must be 2D")
    if (location[0] - radius < 0 or location[0] + radius > dim[0] or location[1] - radius < 0 or location[1] + radius > dim[1]):
        raise Exception("circle is out of bound")
    for i in range(radius):
        for j in range(radius):




def generate_vector():
    # Generate a random vector of size 1000
    vector = np.random.rand(500, 500)

    m = vector.max()
    n = vector.min()

    range = 2 * max(m, n)
    vector = vector/ range

    return vector

folderpath = r"D:\FTLE\FTLE-generated-data\vector-V"

for i in range (100):
    vector = generate_vector()
    file_path = os.path.join(folderpath, f"vector{i}.npy")
    np.save(file_path, vector)
    print(f"vector {i} saved")