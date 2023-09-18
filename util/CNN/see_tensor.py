import torch
from matplotlib import pyplot as plt
import numpy as np

path = r"C:\Users\doubi\Downloads\gt0.pt"
output = torch.load(path)
output = output.cpu().detach().numpy()
output = output[0]
m=output.max
print(np.max(output.max()))
print(np.min(output.min()))
for img in output:
    plt.imshow(img, cmap='gray')
    plt.show()