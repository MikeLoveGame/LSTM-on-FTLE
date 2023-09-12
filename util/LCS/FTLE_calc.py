from vector_field2D import Field
import numpy as np
import os
from matplotlib import pyplot as plt


def readData2D( folderPath : str)-> np.ndarray:
    folder = os.listdir(folderPath)
    data = []
    for file in folder:
        path = os.path.join(folderPath, file)
        temp = np.load(file=path, allow_pickle=True)
        print(f"file {file} loaded")
        data.append(temp)

    data = np.asarray(a=data, dtype=np.float32)
    return data


def showFTLE(ftle : np.ndarray):
    plt.imshow(ftle, cmap='viridis')
    plt.show()


dirPath = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\Training-Set-2\U"
U = readData2D(dirPath)
dirPath = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\Training-Set-2\V"
V = readData2D(dirPath)

shape = U.shape
time = shape[0]
targets = []

finite_time = 60
path = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\Training-Set-2\Targets"

for i in range (0, time):
    dataU = U[i]
    dataV = V[i]
    field = Field(mapU=dataU, mapV=dataV, num_partical=250000, interpolate_factor=10, trace_time=11)
    ftle = field.computeFTLE(time=10)
    targets.append(ftle)
    print(f"ftle at time {i} calculated")


filename = "ftle.npy"
fullpath = os.path.join(path, filename)
np.save(file=fullpath, arr=targets)

