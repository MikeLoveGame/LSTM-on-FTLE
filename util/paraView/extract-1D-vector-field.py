import os
import numpy as np


def to_1D_raw():
    dir_destination=r"D:\AI-Data\Field-Data\0001\result-200\0"
    save_destination = r"C:\AI\Data\Field-data\UV-vect" \
                       r"or-1D"
    files = os.listdir(dir_destination)

    for file in files:
        path = os.path.join(dir_destination, file)
        data = np.fromfile(path, dtype=np.float32)
        try:
            data = data.reshape((50, 500, 500))
        except:
            continue
        name = file.strip(".raw")
        filename = name+"-1D.raw"
        destination_file = os.path.join(save_destination, filename)
        data_1D = data[0]
        data_1D = data_1D.tobytes()
        print(file)
        with open(destination_file, 'wb') as file:
            file.write(data_1D)


def to_1D_numpy():
    dir_destination = r"C:\AI\Data\Field-data\UV-vector-numpy"
    save_destination = r"C:\AI\Data\Field-data\UV"
    files = os.listdir(dir_destination)

    for file in files:
        path = os.path.join(dir_destination, file)
        data = np.load(file=path, allow_pickle=True)
        data_1D=data[0]
        name=file.strip(".npy")
        filename = name+"-1D.npy"
        destination_file = os.path.join(save_destination, filename)
        np.save(file=destination_file,arr=data_1D)
        print(filename)

to_1D_numpy()