'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-16 21:52:04
 '''
from src.DataLoader import CIFAR_10_Test, CIFAR_10_train
import torch
from torch.utils.data import DataLoader
import numpy as np


noise_type = 'clean_label'

train_dataset = CIFAR_10_train(noise_type)

train_dataloader = DataLoader(train_dataset, batch_size= 512)

norm_params = train_dataset.get_norm_params()

test_dataset = CIFAR_10_Test(norm_params)
test_dataloader = DataLoader(test_dataset, batch_size= 512)


model = torch.load('./log/autoencoder/autoencoder_110_0.05368_512dim.pt')
model.eval()


for i, (data, label) in enumerate(train_dataloader):

    data = data.cuda()

    with torch.no_grad():
        if i == 0:
            all_train_feature = model.encoder(data).reshape(-1, 512)
            all_label = label

        else:
            all_train_feature = torch.concat((all_train_feature, model.encoder(data).reshape(-1, 512)), dim = 0)
            all_label = torch.concat([all_label, label], dim = 0)


# print(all_train_feature.shape, all_label.shape)

all_train_feature = all_train_feature.cpu().numpy()
all_label = all_label.numpy()

np.save(f'train_feature.npy', all_train_feature)
np.save(f'all_label.npy', all_label)

# np.save(f'./data/test_data/train_feature.npy', all_train_feature)
# np.save(f'./data/test_data/all_label.npy', all_label)