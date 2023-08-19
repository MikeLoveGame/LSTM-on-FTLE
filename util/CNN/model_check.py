import torch
from matplotlib import pyplot as plt
import numpy as np
from data_loader import test_loaderLSTM
from LSTM import ConvNet
from SSIM import SSIMLoss

def showFTLE(ftle : np.ndarray):
    plt.imshow(ftle, cmap='gray')
    plt.show()



U_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\U-vector-numpy-1D"
V_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\V-vector-numpy-1D"
labelpath = r"D:\FTLE\test-data\targets"

save_folder = r"D:\FTLE\FTLE-generated-data\data-to-see"
model = ConvNet(num_output=9)
device = "cpu"
checkpoint = torch.load(r"D:\FTLE\FTLE-generated-data\best-models\model9\model9.pt", map_location=device)

model.load_state_dict(checkpoint)

resultFile = r"C:\AI\CNIC\SAM\segment-anything\util\CNN\result\model9(9).txt"

model.eval()
model.to(device="cpu")

loss_function = SSIMLoss()

dataset = test_loaderLSTM(1, U_folder, V_folder, labelpath, "cpu")

for i, (inputs, targets) in enumerate(dataset):
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)

    for img in targets[0]:
        showFTLE(img.cpu().detach().numpy())

    outputs = model(inputs)
    outputs = outputs/10
    loss = loss_function(outputs, targets)

    print(f"loss: {loss}")
    for img in outputs[0]:
        showFTLE(img.cpu().detach().numpy())


    torch.save(outputs, save_folder + f"\output{i}.pt")


