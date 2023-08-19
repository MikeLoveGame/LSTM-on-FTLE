from matplotlib import pyplot as plt
import numpy as np


def showFTLE(ftle : np.ndarray):
    ftle = ftle[30]
    i =0
    while(i < ftle.shape[0]) :
        plt.imshow(ftle[i], cmap='gray')
        plt.show()
        i=i+10

path = r"D:\FTLE\test-data\targets\ftle.npy"


ftle = np.load(path, allow_pickle=True)
showFTLE(ftle)

