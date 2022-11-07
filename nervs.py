# 定义卷积神经网络，这里使用5层的神经网络
import torch
import torch.nn as nn
class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3),padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2) )
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2) )
        self.fc1 = nn.Linear(256*7*7, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 6)
def forward(self, img):
        output = self.conv1(img)
        output = self.relu(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool2(output)
        feature = output.view(-1, 256*7*7)
        output = self.fc1(feature)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output