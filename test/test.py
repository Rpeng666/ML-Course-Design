import numpy as np
import torch


test_data = np.load('./cifar-10-100n-main/data/CIFAR-100_human.npy', allow_pickle=True)

# print(test_data)

# test_data = torch.load('./cifar-10-100n-main/data/CIFAR-100_human.pt')

print(test_data)
