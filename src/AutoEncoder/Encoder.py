'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-13 21:35:13
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-13 21:52:19
 '''


import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride= 1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride = 2, padding=1)
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.tanh = nn.Tanh()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride= 1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride= 1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=1, padding= 1)
        self.conv33 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()


        self.conv4 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride= 1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=1, padding= 1)
        self.conv44 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(100352, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024,10)

        self.softmax = nn.Softmax(1)

    def forward(self, X):
        X = self.conv1(X)
        X = self.pool1(X)
        X = self.bn1(X)
        X = self.conv11(X)
        X = self.tanh(X)

        X = self.conv2(X)
        X = self.pool2(X)
        X = self.conv22(X)
        X = self.relu1(X)
        X = self.bn2(X)
        

        X = self.conv3(X)
        X = self.pool3(X)
        X = self.conv33(X)
        X = self.relu2(X)
        X = self.bn3(X)
        
        
        X = self.conv4(X)
        X = self.pool4(X)
        X = self.conv44(X)
        X = self.relu4(X)
        X = self.bn4(X)
        

        X = self.flatten(X)
        X = self.fc1(X)
        X = self.drop1(X)
        X = self.fc2(X)

        X = self.softmax(X)

        return X
        

        


