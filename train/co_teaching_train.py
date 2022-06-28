'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-15 12:50:54
 '''

from src.DataLoader import *
from src.Co_teaching.model import Co_model
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, RandomSampler


co_model_1 = Co_model().cuda()
co_model_2 = Co_model().cuda()


# loss_fn = nn.MSELoss().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
# loss_fn = nn.L1Loss()


batch_size = 400

# optimizer_1 = optim.Adam(co_model_1.parameters())
# optimizer_2 = optim.Adam(co_model_2.parameters())                                                                            
# optimizer = optim.SGD(autoencoder.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9)
optimizer_1 = optim.SGD(co_model_1.parameters(), lr = 0.01)
optimizer_2 = optim.SGD(co_model_2.parameters(), lr = 0.01)    
# optimizer = optim.RMSprop(autoencoder.parameters(), lr = 0.1, alpha=0.9)


train_dataset = CIFAR_10_train('worse_label')
sampler = RandomSampler(train_dataset, replacement = True)
train_dataloader = DataLoader(train_dataset, sampler = sampler, batch_size= batch_size)

norm_params = train_dataset.get_norm_params()

test_dataset = CIFAR_10_Test(norm_params)
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)


all_iter = len(train_dataloader)

all_epoch = 150

log_file = open('./log/co_teaching/history.log', 'w', encoding= 'utf-8')

alpha_plan = [0.1] * 30 + [0.01] * 120

best_acc = 0

def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


for epoch in range(all_epoch):

    co_model_1.train()
    co_model_2.train()

    adjust_learning_rate(optimizer_1, epoch, alpha_plan)
    adjust_learning_rate(optimizer_2, epoch, alpha_plan)

    for i, (data, target) in enumerate(train_dataloader):

        data = data.cuda()     

        if epoch <= 10:
            output_1 = co_model_1(data).cpu()
            output_2 = co_model_2(data).cpu()

            train_acc_1 = (torch.argmax(output_1, dim = 1) == target).sum()/target.shape[0]
            train_acc_2 = (torch.argmax(output_2, dim = 1) == target).sum()/target.shape[0]

            loss_1 = loss_fn(output_1, target)
            loss_2 = loss_fn(output_2, target)

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            loss_1.backward()
            loss_2.backward()

            optimizer_1.step()
            optimizer_2.step()

        else:
            if (i % 2 == 0):
                with torch.no_grad():
                    output = co_model_1(data)
                    index = (output.max(dim = 1)[0] >= 0.8)

                output = co_model_2(data[index]).cpu()

                train_acc_2 = (torch.argmax(output, dim = 1) == target[index]).sum()/target[index].shape[0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

                loss_2 = loss_fn(output, target[index])

                optimizer_2.zero_grad()

                loss_2.backward()

                optimizer_2.step()

            else:
                with torch.no_grad():
                    output = co_model_2(data)
                    index = (output.max(dim = 1)[0] >= 0.8)

                output = co_model_1(data[index]).cpu()

                train_acc_1 = (torch.argmax(output, dim = 1) == target[index]).sum()/target[index].shape[0]

                loss_1 = loss_fn(output, target[index])

                optimizer_1.zero_grad()

                loss_1.backward()

                optimizer_1.step()
            

        if(i %10 == 0):
            print(f'epoch:{epoch: 3}/ {all_epoch} iter: {i:3}/ {all_iter}  |  loss_1:{loss_1: .3f}  acc_1:{train_acc_1 : .3f}  |  loss_2:{loss_2 :.3f}  acc_2:{train_acc_2: .3f}', file = log_file, flush = True)
            print(f'epoch:{epoch: 3}/ {all_epoch} iter: {i:3}/ {all_iter}  |  loss_1:{loss_1: .3f}  acc_1:{train_acc_1 : .3f}  |  loss_2:{loss_2 :.3f}  acc_2:{train_acc_2: .3f}')
    

    co_model_1.eval()
    co_model_2.eval()
    acc_1 = 0
    acc_2 = 0

    with torch.no_grad():
        for j, (data, target) in enumerate(test_dataloader):
            data = data.cuda()

            output_1 = co_model_1(data).cpu()
            output_2 = co_model_2(data).cpu()

            acc_1 += (target == torch.argmax(output_1, dim = 1)).sum()
            acc_2 += (target == torch.argmax(output_2, dim = 1)).sum()

        acc_1 = acc_1/len(test_dataset)
        acc_2 = acc_2/len(test_dataset)

        print(f'test acc_1: {acc_1} acc_2: {acc_2}')


    if( acc_1 > best_acc or acc_2 > best_acc):
        best_acc = max(acc_1, acc_2)

        if(best_acc >= 0.7):
            if(acc_1 > acc_2):
                torch.save(co_model_1 ,f'best_model_{best_acc}.pt')
            else:
                torch.save(co_model_2 ,f'best_model_{best_acc}.pt')


log_file.close()