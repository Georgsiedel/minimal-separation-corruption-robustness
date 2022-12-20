from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tqdm import tqdm
from skimage.util import random_noise
import numpy as np
import re
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
parser.add_argument('--noise', default='gaussian', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.1, type=float, help='perturbation radius')
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
def train(pbar):
    """ Perform epoch of training"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        if args.epsilon != 0:
            if args.noise == 'uniform-linf':
                inputs_dist = dist.Uniform(low=torch.max(inputs - args.epsilon, x_min), high=torch.min(inputs + args.epsilon, x_max))
                inputs_pert = inputs_dist.sample().to(device)
            elif args.noise == 'gaussian':
                var = args.epsilon * args.epsilon
                inputs_pert = torch.tensor(random_noise(inputs, mode='gaussian', mean=0, var=var, clip=True))
            elif 'uniform-l' in args.noise: #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
                inputs_pert = inputs
                for id, img in enumerate(inputs):
                    d = 32*32*3 # number of dimensions of CIFAR-10 image
                    lp = [int(x) for x in re.findall(r'-?\d+\.?\d*', args.noise)] #extract Lp-number from args.noise variable
                    u = np.random.laplace(0, 1/lp, size=(3, 32, 32))  #array of d image-sized Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
                    norm = np.sum(abs(u)**lp)**(1/lp) #scalar, norm samples to lp-norm-sphere
                    r = np.random.random() ** (1.0 / d) #scalar, d-th root to distribute samples from the sphere into the epsilon-size Lp-norm-ball
                    corr = args.epsilon * r * u / norm #, image-sized corruption, epsilon * random radius * random array / normed
                    noisy_img = img + corr #construct corrupted image by adding sampled noise
                    inputs_pert[id] = np.ma.clip(noisy_img, 0, 1) #clip values below 0 and over 1
            else:
                print('Unknown type of noise')
        else:
            inputs_pert = inputs

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
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

        #if args.epsilon != 0:
        #    if args.noise == 'uniform-linf':
        #        inputs_dist = dist.Uniform(low=torch.max(inputs - args.epsilon, x_min), high=torch.min(inputs + args.epsilon, x_max))
        #        inputs_pert = inputs_dist.sample().to(device)
        #    if args.noise == 'gaussian':
        #        var = args.epsilon * args.epsilon
        #        inputs_pert = torch.tensor(random_noise(inputs, mode='gaussian', mean=0, var=var, clip=True))
        #    if args.noise == 'uniform-l2':  # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        #        for img in enumerate(inputs):
        #            d = 32 * 32 * 3  # number of dimensions of CIFAR-10 image
        #            u = np.random.normal(0, 1, size=(3, 32, 32))  # an array of d normally distributed random variables
        #            norm = np.sum(u ** 2) ** (0.5)  # norm gaussian samples onto sphere
        #            r = np.random(0, 1) ** (1.0 / d)  # d-th root to distribute samples from the sphere into the epsilon-size L2-norm-ball
        #            noise = args.epsilon * r * u / norm
        #            inputs_pert = img + noise
        #    else:
        #        print('Unknown type of noise')
        #else:
        inputs_pert = inputs

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
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
        checkpoint = torch.load(f'./experiments/models/{args.noise}/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')
            
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
        torch.save(state, f'./experiments/models/{args.noise}/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')
