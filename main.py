from src.DataLoader import CIFAR_10_Test, CIFAR_10_train
import torch
from PIL import Image
import numpy as np
from random import randint


model = torch.load('./log/autoencoder/autoencoder_110_0.05368_512dim.pt')
torch.save
train_set = CIFAR_10_train('clean_label')

mean, std = train_set.get_norm_params()

def draw_img(temp, file_name):
    temp = temp.reshape(3,32,32)

    img_data = np.zeros(shape=(32,32,3))

    img_data[:,:,0] = temp[0,:,:]*std[0].item() + mean[0].item()
    img_data[:,:,1] = temp[1,:,:]*std[1].item() + mean[1].item()
    img_data[:,:,2] = temp[2,:,:]*std[2].item() + mean[2].item()

    origin_img = Image.fromarray(np.uint8(img_data), mode='RGB')

    origin_img.save(f'{file_name}')

for i in range(5):
    rd_num = randint(0, len(train_set))
    img_tensor, label = train_set[rd_num]

    out_img_tensor = model(img_tensor.reshape(1,3,32,32).cuda()).cpu()

    draw_img(img_tensor, f'{i}_{label}_before.png')

    draw_img(out_img_tensor.detach().numpy(), f'{i}_{label}_after.png')

