import torch
from torch import nn
from data_loader import dataLoaderLSTM
import numpy as np
from SSIM import SSIMLoss

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = "cuda:0"
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias


        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, input_tensor, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(batch_size=input_tensor.size(0), image_size=input_tensor.size()[2:])

        cur_state = hidden_state

        cur_state = self.cell(input_tensor, cur_state)

        return cur_state


class ConvNet(nn.Module):
    def __init__(self, num_output : int = 1):
        super(ConvNet, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=2, hidden_dim=64, kernel_size=(3, 3), bias=True)
        self.device = "cuda:0"

        self.fc_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.num_output = num_output



    def forward(self, x):

        h_next, c_next = self.conv_lstm(x)
        outputs = self.fc_layers(c_next)
        outputs = outputs[0]
        for i in range(1, self.num_output):
            h_next, c_next = self.conv_lstm(x, (h_next, c_next))

            outputs = torch.cat((outputs, self.fc_layers(c_next)[0]))

        outputs = outputs.view(1, 9, 500, 500)

        return outputs



def save_model(model : nn.Module, modelpath = r"D:\FTLE\FTLE-generated-data\best-models\model1.pt" ):
    torch.save(model.state_dict(), modelpath)
def train(modelpath = None):
    device = "cuda:0"
    model = ConvNet(num_output=9)
    model.to(device=device)
    if(modelpath != None):
        checkpoint = torch.load(modelpath, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("load model from " + modelpath)
    learning_rate = 1e-4
    # Set the loss function and the optimizer
    loss_function = SSIMLoss()
    loss_function.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Number of epochs (iterations over the whole dataset)
    epochs = 1000
    MyTrainingSet = dataLoaderLSTM(1, U_folder=r"/home/shike/AI/FTLE/data/U", V_folder=r"/home/shike/AI/FTLE/data/V",
                                   labels_path=r"/home/shike/AI/FTLE/data/FTLE")
    min_loss = []
    for i in range(10):
        min_loss.append(1e4)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(MyTrainingSet):
            # Ensure the data and targets are on the same device as the model
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = loss_function(outputs, targets)
            for i in range(len(min_loss)):
                if loss.item() < min_loss[i]:
                    save_model(model, r"/home/shike/AI/FTLE/model11/model"+str(i)+".pt")
                    min_loss[i] = loss.item()
                    break
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 steps
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

train(modelpath=r"/home/shike/AI/FTLE/model9/model9.pt")
