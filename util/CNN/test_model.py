import torch
from matplotlib import pyplot as plt
import numpy as np
from data_loader import dataLoaderLSTM
from LSTM import ConvNet


def showFTLE(ftle : np.ndarray):
    plt.imshow(ftle, cmap='gray')
    plt.show()



U_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\U-vector-numpy-1D"
V_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\V-vector-numpy-1D"
labelpath = r"D:\FTLE\FTLE-generated-data\targets"

save_folder = r"D:\FTLE\FTLE-generated-data\data-to-see"
learning_rate = 1e-5
model = ConvNet(num_output=9)
checkpoint = torch.load(r"D:\FTLE\FTLE-generated-data\best-models\model1.pt")
model.load_state_dict(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.eval()
model.to(device="cuda")

loss_function = torch.nn.MSELoss()

dataset = dataLoaderLSTM(1, U_folder, V_folder, labelpath)

for i, (inputs, targets) in enumerate(dataset):
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)

    # Forward pass
    outputs = model(inputs)
    torch.save(outputs, save_folder + f"\output{i}.pt")
    # Calculate loss
    loss = loss_function(outputs, targets)
    print(f"loss: {loss}")


