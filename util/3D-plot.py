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
    pd.options.plotting.backend = "plotly"
    z=0
    for data in slices:

        color = np.sum(data[:, :], axis=2 )

        df = pd.DataFrame(color)

        fig = df.plot.scatter(x="a", y="b",)
        fig.show()
        exit(0)


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