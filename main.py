from numpy import dtype
from src.DataLoader import My_DataLoader
from src.AutoEncoder.Encoder import Encoder
import torch.nn as nn
import torch.optim as optim
import torch


encoder = Encoder()

encoder = encoder.cuda()

loss_fn = nn.CrossEntropyLoss().cuda()

optimizer = optim.Adam(encoder.parameters())


train_loader = My_DataLoader().get_data_loader('clean_label', batch_size=256)

for epoch in range(500):

    for i, (data, target) in enumerate(train_loader):
        
        data = data.cuda()
        target = target.long().cuda()

        output = encoder(data)

        acc = (target == torch.argmax(output, dim = 1)).sum()/output.shape[0]

        loss = loss_fn(output, target)

    print(f'epoch:{epoch} iter: {i} loss: {loss} acc: {acc}')


    