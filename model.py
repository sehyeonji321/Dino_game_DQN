# model.py

import torch
import torch.nn as nn

class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.max_pool2d(self.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
