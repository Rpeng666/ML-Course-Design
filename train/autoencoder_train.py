'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-15 12:50:54
 '''

from src.DataLoader import *
from src.AutoEncoder.AutoEncoder import AutoEncoder
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader



autoencoder = AutoEncoder()
autoencoder = autoencoder.cuda()


loss_fn = nn.MSELoss().cuda()

batch_size = 512

optimizer = optim.Adam(autoencoder.parameters())                                                                            
# optimizer = optim.SGD(autoencoder.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
# optimizer = optim.SGD(autoencoder.parameters(), lr = 0.001)
# optimizer = optim.RMSprop(autoencoder.parameters(), lr = 0.1, alpha=0.9)


train_dataset = CIFAR_10_train('clean_label')
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)


all_iter = len(train_dataloader)

all_epoch = 150

log_file = open('./log/autoencoder/history_512dims.log', 'w', encoding= 'utf-8')

for epoch in range(all_epoch):

    autoencoder.train()

    loss_history = []

    for i, (data, target) in enumerate(train_dataloader):
        
        data = data.cuda()         

        output = autoencoder(data)

        loss = loss_fn(output, data)

        loss_history.append(loss)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if(i %10 == 0):
            print(f'epoch:{epoch}/ {all_epoch} iter: {i}/ {all_iter} loss: {loss} ', file = log_file, flush = True)
            print(f'epoch:{epoch}/ {all_epoch} iter: {i}/ {all_iter} loss: {loss} ')
    
    if (epoch >= 100 and epoch %10 == 0):
        aver_loss = sum(loss_history)/len(loss_history)
        torch.save(autoencoder, f'autoencoder_{epoch}_{aver_loss}.pt')

log_file.close()

    