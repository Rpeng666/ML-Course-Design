'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-16 20:54:31
 '''

import torch 
from src.DataLoader import CIFAR_10_Test, CIFAR_10_train
from random import randint
from PIL import Image
import numpy as np



model = torch.load('./log/autoencoder/autoencoder_110_0.05368_512dim.pt')
model.cpu()
model.eval()

train_dataset = CIFAR_10_train('clean_label')

mean, std = train_dataset.get_norm_params()

def draw_img(temp, file_name):
    temp = temp.reshape(3,32,32)

    img_data = np.zeros(shape=(32,32,3))

    img_data[:,:,0] = temp[0,:,:]*std[0].item() + mean[0].item()
    img_data[:,:,1] = temp[1,:,:]*std[1].item() + mean[1].item()
    img_data[:,:,2] = temp[2,:,:]*std[2].item() + mean[2].item()


    origin_img = Image.fromarray(np.uint8(img_data), mode='RGB')

    origin_img.save(f'{file_name}')


test_dataset = CIFAR_10_Test((mean, std))

random_num = randint(0, len(test_dataset))

temp, label = test_dataset[random_num]


draw_img(temp.numpy(), 'origin_img.png')


#``````````````````````````````#

temp = model(temp.reshape((1,3,32,32)))

temp = temp.reshape(3,32,32)

draw_img(temp.detach().numpy(),'new_img.png')