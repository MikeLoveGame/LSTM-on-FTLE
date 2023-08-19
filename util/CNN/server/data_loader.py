import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

class MyDataset(Dataset):
    def __init__(self, U_path, V_path, labels_path):
        self.device = "cuda"
        # Load data from numpy files
        data_U = __readData2D__(U_path)
        data_V = __readData2D__(V_path)
        data = [data_U, data_V]
        self.data = []
        self.labels = np.load(labels_path)
        for i in range (len(self.labels)):
            j = i+1
            set = []
            frame1 = [data[0][i], data[1][i]]
            frame2 = [data[0][j], data[1][j]]
            set.append(frame1)
            set.append(frame2)
            self.data.append(set)
        self.data = np.asarray(a=self.data, dtype=np.float32)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

        self.data.cuda()
        self.labels.cuda()

    def __getitem__(self, index):
        # Get one item from the dataset
        U = self.data[0]
        V = self.data[1]
        data = torch.cat(tensors=[U, V], dim=0)
        data = data[index]
        label = self.labels[index]


        return data, label

    def __len__(self):
        # Return the total number of data samples
        return len(self.labels)

class LSTM_DataSet2D(Dataset):
    def __init__(self, U_path, V_path, labels_path, device):
        self.device = "cuda:0"
        # Load data from numpy files
        data_U = __readData2D__(U_path)
        data_V = __readData2D__(V_path)
        data = [data_U, data_V]
        self.data = []
        self.labels = np.load(labels_path)
        for i in range (len(self.labels)):
            j = i+1
            set = []
            frame1 = [data[0][i], data[1][i]]
            frame2 = [data[0][j], data[1][j]]
            set.append(frame1)
            set.append(frame2)
            self.data.append(set)
        self.data = np.asarray(a=self.data, dtype=np.float32)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        if (device == "cuda"):
            self.data.cuda()
            self.labels.cuda()
class LSTM_DataSet(Dataset):
    def __init__(self, U_path, V_path, labels_path, device="cuda"):
        self.device = "cuda:0"
        # Load data from numpy files
        data_U = __readData2D__(U_path)
        data_V = __readData2D__(V_path)
        self.normalizationVector(data_U, data_V)

        data = []
        time = data_U.shape[0]
        for i in range(time):
            frame = [data_U[i], data_V[i]]
            data.append(frame)

        labels = __readData2D__(labels_path)
        labels = labels[0]

        self.labels = labels

        self.data = np.asarray(a=data, dtype=np.float32)


        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

        if (device == "cuda"):
            self.data.cuda()
            self.labels.cuda()

    def __getitem__(self, index):
        # Get one item from the dataset
        data = self.data[index]
        label = self.labels[index]

        return data, label

    def __len__(self):
        # Return the total number of data samples
        return len(self.labels)
    @staticmethod
    def normalizationVector(U : np.ndarray, V : np.ndarray):
        shapeU = U.shape
        shapeV = V.shape

        maxU = U.max()
        minU = U.min()
        maxV = V.max()
        minV = V.min()

        m = max(maxU, maxV)
        n = min(minU, minV)
        if (m == n):
            return
        for i in range(shapeU[0]):
            for j in range(shapeU[1]):
                for k in range(shapeU[2]):

                    U[i][j][k] = (U[i][j][k] - n) / (m - n)

    @staticmethod
    def normalization(data : np.ndarray):
        shape = data.shape
        max = data.max()
        min = data.min()
        print(f"max = {max}, min = {min}")
        if (max == min):
            return
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):

                    data[i][j][k] = (data[i][j][k] - min) / (max-min)



def __readData2D__(folderPath : str)-> np.ndarray:
    folder = os.listdir(folderPath)
    data = []
    for file in folder:
        path = os.path.join(folderPath, file)
        temp = np.load(file=path, allow_pickle=True)
        print(f"file {file} loaded")
        data.append(temp)

    data = np.asarray(a=data, dtype=np.float32)
    return data

def dataLoaderLSTM(batch_size : int, U_folder = r"D:\FTLE\FTLE-generated-data\vector-U" ,
                   V_folder = r"D:\FTLE\FTLE-generated-data\vector-V" ,
                   labels_path = r"D:\FTLE\FTLE-generated-data\targets"):

    # Create the Dataset
    dataset = LSTM_DataSet(U_path=U_folder, V_path=V_folder, labels_path=labels_path)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
def test_loaderLSTM(batch_size : int, U_folder, V_folder, labels_path, device):
    dataset = LSTM_DataSet(U_path=U_folder, V_path=V_folder, labels_path=labels_path, device=device)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
def dataLoaderCNN():
    # Set the paths to the numpy files
    U_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\U-vector-numpy-1D"
    V_folder = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\V-vector-numpy-1D"

    labels_path = r"C:\AI\CNIC\SAM\segment-anything\util\LCS\data\ftle.npy"

    # Create the Dataset
    dataset = MyDataset(U_path=U_folder, V_path=V_folder, labels_path=labels_path)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    return dataloader
def test():
    U_folder = r"D:\FTLE\FTLE-generated-data\vector-U"
    V_folder = r"D:\FTLE\FTLE-generated-data\vector-V"

    labels_path = r"D:\FTLE\FTLE-generated-data\targets"

    # Create the Dataset
    dataset = LSTM_DataSet(U_path=U_folder, V_path=V_folder, labels_path=labels_path)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

