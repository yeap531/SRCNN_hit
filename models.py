from torch import nn
import torch.nn.functional as F

import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        return out