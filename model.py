import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
device = torch.device("cpu")

n_in, n_layer_1, n_layer_2, n_out = 784, 64, 32, 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


net = Net()
