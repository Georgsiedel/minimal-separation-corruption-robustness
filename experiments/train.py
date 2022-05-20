from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tqdm import tqdm
import os
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributions as dist
import torchvision
import torchvision.transforms as transforms

from experiments.network import WideResNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with perturbations')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epsilon', default=0.157, type=float, help='perturbation radius')
parser.add_argument('--epochs', default=30, type=int, help="number of epochs")
parser.add_argument('--run', default=0, type=int, help='run number')
args = parser.parse_args()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
criterion = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Bounds for corruption training input
#x_min = torch.tensor([(0.0 - 0.4914)/0.2023, (0.0 - 0.4822)/0.1994, (0.0 - 0.4465)/0.2010]).to(device).view([1, -1, 1, 1])
#x_max = torch.tensor([(1.0 - 0.4914)/0.2023, (1.0 - 0.4822)/0.1994, (1.0 - 0.4465)/0.2010]).to(device).view([1, -1, 1, 1])
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])
print(os.getcwd())
def train(pbar):
    """ Perform epoch of training"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.epsilon != 0:
            inputs_dist = dist.Uniform(low=torch.max(inputs - args.epsilon, x_min), high=torch.min(inputs + args.epsilon, x_max))
            inputs_pert = inputs_dist.sample().to(device)
        else:
            inputs_pert = inputs

        targets_pert = targets
        targets_pert_pred = net(inputs_pert)
        
        loss = criterion(targets_pert_pred, targets_pert)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        pbar.update(1)

    train_acc = 100.*correct/total
    return train_acc

def test(pbar):
    """ Test current network on test set"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if args.epsilon != 0:
            inputs_dist = dist.Uniform(low=torch.max(inputs - args.epsilon, x_min), high=torch.min(inputs + args.epsilon, x_max))
            inputs_pert = inputs_dist.sample().to(device)
        else:
            inputs_pert = inputs

        targets_pert = targets
        targets_pert_pred = net(inputs_pert)
        
        loss = criterion(targets_pert_pred, targets_pert)

        test_loss += loss.item()
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

        pbar.set_description('[Test] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        pbar.update(1)

    acc = 100.*correct/total
    return acc

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./experiments/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)    #, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./experiments/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False) #, num_workers=2)

    # Construct model
    print('\nBuilding model..')
    net = WideResNet(28, 10, 0.3, 10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('\nResuming from checkpoint..')
        checkpoint = torch.load(f'./experiments/models/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')
            
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1

    # Number of batches
    # NOTE: It's 50,000/32 + 10,000/10
    total_steps = (1563 + 1000) * args.epochs

    # Training loop
    print('\nTraining model..')
    with tqdm(total=total_steps) as pbar:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            train_acc = train(pbar)
            acc = test(pbar)

        # Save final epoch
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'train_acc': train_acc,
            'epoch': start_epoch+args.epochs-1,
        }  
        torch.save(state, f'./experiments/models/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')
