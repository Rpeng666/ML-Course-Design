'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-15 12:50:54
 '''

from src.DataLoader import *
from src.AutoEncoder.Encoder import Encoder
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

encoder = Encoder()

encoder = encoder.cuda()

loss_fn = nn.CrossEntropyLoss().cuda()

batch_size = 256

# optimizer = optim.Adam(encoder.parameters())                                                                            
optimizer = optim.SGD(encoder.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
# optimizer = optim.SGD(encoder.parameters(), lr = 0.01)
# optimizer = optim.RMSprop(encoder.parameters(), lr = 0.1, alpha=0.9)


train_dataset = CIFAR_10_train('clean_label')
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)

norm_params = train_dataset.get_norm_params()

test_dataset = CIFAR_10_Test(norm_params)
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)


all_iter = len(train_dataloader)

all_epoch = 500

log_file = open('./log/encoder/history.log', 'w', encoding= 'utf-8')

for epoch in range(all_epoch):

    encoder.train()

    for i, (data, target) in enumerate(train_dataloader):
        
        data = data.cuda()
        target = target.long().cuda()          

        output = encoder(data)

        acc = (target == torch.argmax(output, dim = 1)).sum()/output.shape[0]

        loss = loss_fn(output, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if(i %10 == 0):
            print(f'epoch:{epoch}/ {all_epoch} iter: {i}/ {all_iter} loss: {loss} acc: {acc}', file = log_file, flush = True)
            print(f'epoch:{epoch}/ {all_epoch} iter: {i}/ {all_iter} loss: {loss} acc: {acc}')

    encoder.eval()

    with torch.no_grad():

        acc_count = 0

        for i, (data, target) in enumerate(test_dataloader):
            data = data.cuda()
            target = target.long().cuda()

            output = encoder(data)

            acc_count += (target == torch.argmax(output, dim = 1)).sum().item()

        acc = acc_count/(len(test_dataloader)*batch_size)

        print(f'Test Acc: {acc}', file = log_file, flush = True)
        print(f'Test Acc: {acc}')

        if(acc > 0.8):
            torch.save(encoder, f'encoder_{acc}_{epoch}.pt')

log_file.close()

    