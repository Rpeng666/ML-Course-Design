'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-13 21:58:09
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-15 13:00:20
 '''

import torch 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import numpy as np
from torchvision import transforms


class CIFAR_10_train(Dataset):
    def __init__(self, noise_type: str) -> None:
        '''获取所有的data,前3072列是特征,
        后面的列分别是:
        clean_label标签,
        aggre_label标签,
        worse_label标签,
        random_label1标签,
        random_label2标签,
        random_label3标签
        '''
                
        for i in range(1,6):
            with open(f'./origin_data/data_batch_{i}', 'rb') as file:
                data = pickle.load(file, encoding='bytes')

                temp = data[b'data']

                if i == 1:
                    self.train_data = temp
                else:
                    self.train_data = np.concatenate([self.train_data, temp])

        self.all_noise_label = torch.load('./baseline_ResNet/data/CIFAR-10_human.pt')
        self.train_data = torch.Tensor(self.train_data)

        self.train_data = self.vector2img(self.train_data)

        self.mean = torch.zeros(size=(3,))
        self.std = torch.zeros(size=(3,))

        for i in range(3):
            self.mean[i] = self.train_data[:, i, :, :].mean()
            self.std[i] = self.train_data[:, i, : , :].std()

        self.labels = torch.Tensor(self.all_noise_label[f'{noise_type}']).long()

        self.transform = transforms.Compose([
            transforms.Normalize(self.mean, self.std),
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip()
        ])

    def __getitem__(self, index: int):

        if self.transform:

            new_train_data = self.transform(self.train_data[index])

            return new_train_data, self.labels[index]
        
        return self.train_data[index], self.labels[index]


    def __len__(self):
        
        return self.train_data.shape[0]


    def vector2img(self, all_img_tensor):

        result = torch.zeros(size=(all_img_tensor.shape[0], 3, 32, 32))

        for index, img_tensor in enumerate(all_img_tensor):

            img_tensor = img_tensor.reshape(-1,1024)

            r = img_tensor[0,:].reshape(32,32)
            g = img_tensor[1,:].reshape(32,32)
            b = img_tensor[2,:].reshape(32,32)

            result[index][0] = r
            result[index][1] = g
            result[index][2] = b

        return result

    
    def get_norm_params(self):

        return self.mean, self.std

        
class CIFAR_10_Test(Dataset):
    def __init__(self, norm_params) -> None:

        self.mean, self.std = norm_params

        with open('./origin_data/test_batch', 'rb') as file:
            data = pickle.load(file, encoding='bytes')

            self.test_data = data[b'data']

            self.label = np.array(data[b'labels']).reshape(-1, 1)

            self.test_data = torch.Tensor(self.test_data)

            self.label = torch.Tensor(self.label).reshape(-1).long()

            self.test_data = self.vector2img(self.test_data)

            for i in range(3):
                self.test_data[:,i,:,:] = (self.test_data[:,i,:,:] - self.mean[i])/self.std[i]


    def __getitem__(self, index: int) :
        return self.test_data[index], self.label[index]


    def __len__(self):

        return self.test_data.shape[0]


    def vector2img(self, all_img_tensor):

        result = torch.zeros(size=(all_img_tensor.shape[0], 3, 32, 32))

        for index, img_tensor in enumerate(all_img_tensor):

            img_tensor = img_tensor.reshape(-1,1024)

            r = img_tensor[0,:].reshape(32,32)
            g = img_tensor[1,:].reshape(32,32)
            b = img_tensor[2,:].reshape(32,32)

            result[index][0] = r
            result[index][1] = g
            result[index][2] = b

        return result