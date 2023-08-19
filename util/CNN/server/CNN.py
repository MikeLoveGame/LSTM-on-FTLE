import torch
from torch import nn
from data_loader import dataLoaderCNN
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(1, 1), #figure this out later
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        return x

model = CNN()


device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device=device)
learning_rate = 1e-6

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

epochs = 10
MyTrainingSet = dataLoaderCNN()

for epoch in range(epochs):
    for i, (inputs, targets) in enumerate(MyTrainingSet):
        # Ensure the data and targets are on the same device as the model
        inputs = inputs.to(device)

        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = loss_function(outputs, targets)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 steps
        if i % 10 == 0:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
