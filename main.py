from numpy import dtype
from src.DataLoader import My_DataLoader
from src.AutoEncoder.Encoder import Encoder
import torch.nn as nn
import torch.optim as optim
import torch


encoder = Encoder()

encoder = encoder.cuda()

loss_fn = nn.CrossEntropyLoss().cuda()

# optimizer = optim.Adam(encoder.parameters())                                                                            
optimizer = optim.SGD(encoder.parameters(), lr=0.001)

my_loder = My_DataLoader()

batch_size = 128

train_loader = my_loder.get_traindata_loader('clean_label', batch_size=batch_size)

test_loader = my_loder.get_testdata_loader(batch_size= batch_size)

all_iter = len(train_loader)

all_epoch = 500

for epoch in range(all_epoch):

    encoder.train()

    for i, (data, target) in enumerate(train_loader):
        
        data = data.cuda()
        target = target.long().cuda()                 

        output = encoder(data)

        acc = (target == torch.argmax(output, dim = 1)).sum()/output.shape[0]

        loss = loss_fn(output, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if(i %10 == 0):
            print(f'epoch:{epoch}/ {all_epoch} iter: {i}/ {all_iter} loss: {loss} acc: {acc}')

    encoder.eval()

    with torch.no_grad():

        acc_count = 0

        for i, (data, target) in enumerate(test_loader):
            data = data.cuda()
            target = target.long().cuda()

            output = encoder(data)

            acc_count += (target == torch.argmax(output, dim = 1)).sum().item()

            print((target == torch.argmax(output, dim = 1)).sum())

        print('debug: ', acc_count, len(test_loader))

        acc = acc_count/(len(test_loader)*batch_size)

        print(f'Test Acc: {acc}')

    