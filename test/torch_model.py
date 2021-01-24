import torch.nn as nn
import torch.nn.functional as F

# create model

d_in = 2
d_out = 1

d_h1 = 3
d_h2 = 4


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_h1)
        self.fc2 = nn.Linear(d_h1, d_h2)
        self.fc3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
