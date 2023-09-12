from matplotlib import pyplot as plt
import numpy as np


def showFTLE(ftle : np.ndarray):
    ftle = ftle[30]
    i =0
    while(i < ftle.shape[0]) :
        plt.imshow(ftle[i], cmap='gray')
        plt.show()
        i=i+10

path = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\Training-Set-2\Targets\ftle.npy"


ftle = np.load(path, allow_pickle=True)
showFTLE(ftle)

