# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):

        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total += 1
        train_correct += prec

        loss = F.cross_entropy(logits, labels, reduction='mean')

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch+1, n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    train_acc = float(train_correct)/float(train_total)

    return train_acc


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0

    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()

    acc = 100*float(correct)/float(total)

    return acc


##################################### main code ################################################


# Hyper Parameters
batch_size = 128
learning_rate = 0.001
dataset = 'cifar10'
noise_type = 'clean'
noise_path = './data/CIFAR-100_human.pt'
n_epoch = 100
best_acc = 0




train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(dataset, noise_type, noise_path)


print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])


# load model
model = ResNet34(num_classes)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

alpha_plan = [0.1] * 60 + [0.01] * 40

model.cuda()


epoch = 0
train_acc = 0

# training

for epoch in range(n_epoch):
    # train models
    print(f'epoch {epoch + 1}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)

    model.train()

    train_acc = train(epoch, train_loader, model, optimizer)

    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)

    if (test_acc > best_acc):
        best_acc = test_acc
        torch.save(model, f'./baseline_ResNet/history/model/model_epoch_{epoch}_acc_{best_acc}.pt')
        
    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
