import torch
import torch.nn as nn
from data_loader import dataLoaderLSTM

class ImageSequenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImageSequenceLSTM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -500:])
        return out



def lossfunction(output, target):
    shape = output.shape
    new_out = []
    if (len(shape) != 3):
        return
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                    loss.append(output[i][j][k] - target[i][j][k])

device = "cuda"

# Number of epochs (iterations over the whole dataset)
epochs = 1000
MyTrainingSet = dataLoaderLSTM(1)

model = ImageSequenceLSTM( input_size= 500, hidden_size= 10, output_size= 500)
model.to(device=device)
learning_rate = 1e-6
# Set the loss function and the optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(MyTrainingSet):
        # Ensure the data and targets are on the same device as the model
        inputs = inputs[0]
        inputs = inputs.to(model.device)
        targets = targets.to(model.device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print loss every 10 steps
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
