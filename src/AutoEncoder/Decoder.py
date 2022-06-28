'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-13 21:35:18
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-13 21:54:06
 '''


import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.convtrans1 = nn.ConvTranspose2d(in_channels=512, out_channels= 128, kernel_size= 4,stride=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128) 

        self.convtrans2 = nn.ConvTranspose2d(in_channels=128, out_channels= 80, kernel_size= 4,stride=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(80)

        self.convtrans3 = nn.ConvTranspose2d(in_channels=80, out_channels= 64, kernel_size= 4,stride=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)

        self.convtrans4 = nn.ConvTranspose2d(in_channels=64, out_channels= 40, kernel_size= 4,stride=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(40)

        self.convtrans5 = nn.ConvTranspose2d(in_channels=40, out_channels= 24, kernel_size= 4,stride=2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(24)

        self.convtrans6 = nn.ConvTranspose2d(in_channels=24, out_channels= 16, kernel_size= 4,stride=1)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(16)

        self.convtrans7 = nn.ConvTranspose2d(in_channels=16, out_channels= 3, kernel_size= 2,stride=1)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(3)


    def forward(self, X):
        X = self.convtrans1(X)
        X = self.bn1(X)
        

        X = self.convtrans2(X)
        X = self.relu2(X)
        X = self.bn2(X)


        X = self.convtrans3(X)
        X = self.relu3(X)
        X = self.bn3(X)


        X = self.convtrans4(X)
        X = self.relu4(X)
        X = self.bn4(X)


        X = self.convtrans5(X)
        X = self.relu5(X)
        X = self.bn5(X)


        X = self.convtrans6(X)
        X = self.relu6(X)
        X = self.bn6(X)


        X = self.convtrans7(X)
        X = self.relu7(X)
        X = self.bn7(X)

        return X

