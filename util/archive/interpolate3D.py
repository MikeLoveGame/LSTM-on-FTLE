import helper_Functions as hf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def show3D(slices : np.ndarray):
    plt.rcParams["figure.figsize"] = [10.200, 10.200]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    z=0
    for data in slices:
        z=z+1
        x = []
        y=[]
        count=0

        for i in range (data.particle_shape[0]):
            x.append(np.ones(shape = [1, data.particle_shape[0]]) * count)
            y.append(np.arange(data.particle_shape[1]))
            count=count+1
        color = np.sum(data[:, :], axis=2 )

        print(f"currently plotting {z} ")
        ax.scatter3D(xs = x, ys = y, zs = z, cmap= color)

    plt.show()

source=r"C:\AI\CNIC\SAM\Data\temp"
img_files = os.listdir(source)
data=[]
for img in img_files:
    img_path = os.path.join(source, img)
    img_data=cv2.imread(img_path)
    img_data=cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    data.append(img_data)

data = np.asarray(data)

show3D(data)