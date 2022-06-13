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

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride= 1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride = 1)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.tanh

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride= 1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride= 1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(18432, 10)
        self.softmax = nn.Softmax(1)

    def forward(self, X):

        X = self.cnn1(X)
        X = self.pool1(X)
        X = self.batch_norm1(X)
        X = self.cnn2(X)
        X = self.pool2(X)
        X = self.batch_norm2(X)
        X = self.cnn3(X)
        X = self.pool3(X)
        X = self.batch_norm3(X)
        X = self.flatten(X)
        X = self.linear(X)
        X = self.softmax(X)

        return X
        

        


