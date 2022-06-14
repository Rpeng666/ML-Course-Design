from dataclasses import dataclass
from matplotlib.pyplot import axis
import torch 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import numpy as np
from torchvision import transforms



class My_DataLoader:
    def __init__(self) -> None:
        self.train_data = np.array([])

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

                label = np.array(data[b'labels']).reshape(-1,1)
                
                temp = np.concatenate([temp, label], axis =1)

                if i == 1:
                    self.train_data = temp
                else:
                    self.train_data = np.concatenate([self.train_data, temp])

        self.noise_label = torch.load('./baseline_ResNet/data/CIFAR-10_human.pt')
        self.train_data = torch.Tensor(self.train_data[:,:3072])

        self.img_matrix = self.vector2img(self.train_data)

        self.mean = torch.zeros(size=(3,))
        self.std = torch.zeros(size=(3,))

        for i in range(3):
            self.mean[i] = self.img_matrix[:, i, :, :].mean()
            self.std[i] = self.img_matrix[:, i, : , :].std()

        for i in range(3):
            self.img_matrix[:,i,:,:] = (self.img_matrix[:,i,:,:] - self.mean[i])/self.std[i]



    def get_traindata_loader(self, noise_type: str, batch_size):
        
        if noise_type == 'clean_label':

            clean_label = torch.Tensor(self.noise_label['clean_label'])

            data_set = TensorDataset(self.img_matrix, clean_label)

            return DataLoader(data_set, batch_size=batch_size, shuffle= True)


        elif noise_type == 'aggre_label':

            aggre_label = torch.Tensor(self.noise_label['aggre_label'])

            data_set = TensorDataset(self.img_matrix, aggre_label)
            return DataLoader(data_set, batch_size=batch_size, shuffle= True)


        elif noise_type == 'worse_label':

            worse_label = torch.Tensor(self.noise_label['worse_label'])

            data_set = TensorDataset(self.img_matrix, worse_label)
            return DataLoader(data_set, batch_size=batch_size, shuffle= True)


        elif noise_type == 'random_label1':

            random_label1 = torch.Tensor(self.noise_label['random_label1'])

            data_set = TensorDataset(self.img_matrix, random_label1)
            return DataLoader(data_set, batch_size=batch_size, shuffle= True)


        elif noise_type == 'random_label2':

            random_label2 = torch.Tensor(self.noise_label['random_label2'])

            data_set = TensorDataset(self.img_matrix, random_label2)
            return DataLoader(data_set, batch_size=batch_size, shuffle= True)

        
        elif noise_type == 'random_label3':

            random_label3 = torch.Tensor(self.noise_label['random_label3'])

            data_set = TensorDataset(self.img_matrix, random_label3)
            return DataLoader(data_set, batch_size=batch_size, shuffle= True)


    def get_testdata_loader(self, batch_size):

        with open('./origin_data/test_batch', 'rb') as file:
            data = pickle.load(file, encoding='bytes')

            temp = data[b'data']

            label = np.array(data[b'labels']).reshape(-1, 1)

            temp = torch.Tensor(temp)

            label = torch.Tensor(label).reshape(-1)

            gray_img = self.vector2img(temp)

            for i in range(3):
                gray_img[:,i,:,:] = (gray_img[:,i,:,:] - self.mean[i])/self.std[i]

            test_set = TensorDataset(gray_img, label)

            test_loader = DataLoader(test_set, batch_size= batch_size, shuffle= True)

            return test_loader


    def get_norm_params(self):

        return self.mean, self.std


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