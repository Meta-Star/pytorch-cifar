'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import copy
import argparse
import numpy as np
from collections import OrderedDict

from models import *
from utils import progress_bar


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/1.0, dim=1)
    softmax_targets = F.softmax(targets/1.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lambda_KD', default=0.5, type=float, help='lambda_KD')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
#net = net.to(device)
labelGenerator = FPN(BasicBlock, 100)
labelGenerator = labelGenerator.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
meta_optimizer = optim.Adam(labelGenerator.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
lr_milestones = [80, 140]
res_lr = 0.1

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    lr = args.lr * (0.1 ** np.sum(epoch >= np.array(lr_milestones)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    res_lr = lr

    meta_lr = 0.01 * (0.1 ** np.sum(epoch >= np.array(lr_milestones)))
    for param_group in meta_optimizer.param_groups:
        param_group['lr'] = meta_lr


    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, outfeat = net(inputs)
        #outputs, feat_loss = net(inputs, baseline=True)
        optimizer.zero_grad()

        loss = criterion(outputs[-1], targets)
        
        #model_learned loss
        outfeat4 = outfeat[0].detach()
        outfeat3 = outfeat[1].detach()
        outfeat2 = outfeat[2].detach()
        outfeat1 = outfeat[3].detach()
        with torch.no_grad():
            label = labelGenerator(outfeat4, outfeat3, outfeat2, outfeat1)
        teacher_labels = []
        for index in range(len(label)):
            teacher_label = label[index].detach()
            teacher_labels.append(teacher_label)
        
        
        """
        #baseline
        teacher_label = outputs[-1].detach()
        teacher_label.requires_grad = False
        """

        """
        #progressive
        teacher_labels = []
        for index in range(1, len(outputs)):
            teacher_label = outputs[index].detach()
            teacher_label.requires_grad = False
            teacher_labels.append(teacher_label)
        """
        for index in range(0, len(outputs)-1):
            loss += criterion(outputs[index], targets) * (1 - args.lambda_KD)
            loss += CrossEntropy(outputs[index], teacher_labels[index]) * args.lambda_KD * 1.0
            #loss += CrossEntropy(outputs[index], teacher_label) * args.lambda_KD * 9.0

        #loss += feat_loss * 5e-7
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs[-1].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.3f'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))

    
    #meta_update
    if epoch%5 == 0:
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, outfeat = net(inputs)
            meta_optimizer.zero_grad()
            
            outfeat4 = outfeat[0].detach()
            outfeat3 = outfeat[1].detach()
            outfeat2 = outfeat[2].detach()
            outfeat1 = outfeat[3].detach()
            label = labelGenerator(outfeat4, outfeat3, outfeat2, outfeat1)
            teacher_labels = []
            for index in range(len(label)):
                teacher_label = label[index].detach()
                teacher_labels.append(teacher_label)       

            loss1 = criterion(outputs[-1], targets)
            loss2 = torch.tensor(0.0).to(device)
            loss3 = torch.tensor(0.0).to(device)
            for index in range(0, len(outputs)-1):
                loss2 += criterion(outputs[index], targets) * (1 - args.lambda_KD)
                loss3 += CrossEntropy(outputs[index], teacher_labels[index]) * args.lambda_KD

            fast_weights = OrderedDict((name, param) for (name, param) in net.named_parameters())

            grads1 = torch.autograd.grad(loss1, net.parameters(), retain_graph=True, allow_unused=True)
            grads2 = torch.autograd.grad(loss2, net.parameters(), retain_graph=True, allow_unused=True)
            grads3 = torch.autograd.grad(loss3, net.parameters(), create_graph=True, allow_unused=True)

            fast_weights = OrderedDict((name, param - res_lr * ((grad1 if (grad1 is not None) else 0)+(grad2 if (grad2 is not None) else 0)+(grad3 if (grad3 is not None) else 0))) for ((name, param), grad1, grad2, grad3) in zip(fast_weights.items(), grads1, grads2, grads3))
            output = net(inputs, fast_weights)
            loss = criterion(output, targets)
            loss.backward()

            meta_optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.3f'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))
    

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, train=False)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
