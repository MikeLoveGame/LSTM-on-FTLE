import torch
from matplotlib import pyplot as plt
import numpy as np


path = r"D:\FTLE\FTLE-generated-data\data-to-see\out1"
output = torch.load(path)
output = output.cpu().detach().numpy()
output = output[0]

for img in output:
    plt.imshow(img, cmap='gray')
    plt.show()