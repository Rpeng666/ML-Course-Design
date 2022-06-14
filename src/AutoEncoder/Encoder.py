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


class BaseBlock(nn.Module):
    def __init__(self, 
    conv_in_channels, 
    conv_out_channels, 
    conv_kernel_size,
    conv_padding,
    conv_stride,
    pool_kernel_size,
    pool_stride,
    pool_padding ) -> None:
        super(BaseBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels= conv_in_channels, 
                out_channels= conv_out_channels, 
                kernel_size= conv_kernel_size, 
                padding= conv_padding, 
                stride= conv_stride
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels= conv_out_channels, 
                out_channels= conv_out_channels, 
                kernel_size= conv_kernel_size, 
                padding= 1, 
                stride= 1
            ),

            nn.MaxPool2d(kernel_size= pool_kernel_size, stride = pool_stride, padding = pool_padding)
        )

    def forward(self, X):
        
        return self.block(X)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        '''conv_in_channels, conv_out_channels, conv_kernel_size, conv_padding, conv_stride, 
        pool_kernel_size, pool_stride, pool_padding'''
        self.conv_pool1 = BaseBlock(3, 32, 4, 0, 1, 3, 1, 0)

        self.conv_pool2 = BaseBlock(32, 64, 4, 0, 1, 3, 1, 0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv_pool3 = BaseBlock(64, 128, 4, 0, 1, 3, 1, 0)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_pool4 = BaseBlock(128, 256, 4, 0, 1, 3, 1, 0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16384, 516)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(516,64)
        self.drop = nn.Dropout()
        self.fc3 = nn.Linear(64, 10)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):

        X = self.conv_pool1(X)

        X = self.conv_pool2(X)
        X = self.bn2(X)

        X = self.conv_pool3(X)
        X = self.bn3(X)

        X = self.conv_pool4(X)

        X = self.flatten(X)

        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.drop(X)
        X = self.fc3(X)
        X = self.softmax(X)

        

        return X
        

        


