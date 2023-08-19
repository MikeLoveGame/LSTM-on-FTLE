import torch
from matplotlib import pyplot as plt
import numpy as np
from data_loader import test_loaderLSTM
from LSTM import ConvNet
import os

def showFTLE(ftle : np.ndarray):
    plt.imshow(ftle, cmap='gray')
    plt.show()



U_folder = r"/home/shike/AI/FTLE/test/U"
V_folder = r"/home/shike/AI/FTLE/test/V"
labelpath = r"/home/shike/AI/FTLE/test/labels"

logfolder = r"/home/shike/AI/FTLE/model4"
model_name = "model4.pt"
model_folder = r"/home/shike/AI/FTLE/model4"
model_path = os.path.join(model_folder, model_name)
learning_rate = 1e-5
model = ConvNet(num_output=9)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.eval()
model.to(device="cpu")

loss_function = torch.nn.MSELoss()

dataset = test_loaderLSTM(1, U_folder, V_folder, labelpath, "cpu")

log= open(os.path.join(logfolder, model_name+"log.txt"), "w")

for i, (inputs, targets) in enumerate(dataset):
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)

    # Forward pass
    outputs = model(inputs)
    # Calculate loss
    loss = loss_function(outputs, targets)
    log.write(f"on {i} the loss is: {loss}\n")

    print(f"loss: {loss}")

log.close()
