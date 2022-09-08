from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.chdir('C:\\Users\\Admin\\Desktop\\Python\\MSCR')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from skimage.util import random_noise
import torch.distributions as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from experiments.network import WideResNet

criterion = nn.CrossEntropyLoss()
# Bounds for corruption evaluation input
#x_min = torch.tensor([(0.0 - 0.4914)/0.2023, (0.0 - 0.4822)/0.1994, (0.0 - 0.4465)/0.2010]).to(device).view([1, -1, 1, 1])
#x_max = torch.tensor([(1.0 - 0.4914)/0.2023, (1.0 - 0.4822)/0.1994, (1.0 - 0.4465)/0.2010]).to(device).view([1, -1, 1, 1])
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])

def compute_metric(loader, net, epsilon, noise):
    """ Calculate A-TSRM"""
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if epsilon != 0:
            if noise == str('uniform-linf'):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs_dist = dist.Uniform(low=torch.max(inputs - epsilon, x_min),
                                           high=torch.min(inputs + epsilon, x_max))
                inputs_pert = inputs_dist.sample().to(device)
            elif noise == str('gaussian'):
                var = epsilon * epsilon
                inputs_pert = torch.tensor(random_noise(inputs, mode='gaussian', mean=0, var=var, clip=True))
            elif noise == str('uniform-l2'):
                inputs_pert = inputs
                for id, img in enumerate(inputs):
                    d = 32*32*3 # number of dimensions of CIFAR-10 image
                    u = np.random.normal(0, 1, size=(3, 32, 32))  #an array of d normally distributed random variables
                    norm = np.sum(u ** 2) ** (0.5) #norm gaussian samples onto sphere
                    r = np.random.random() ** (1.0 / d) #d-th root to distribute samples from the sphere into the epsilon-size L2-norm-ball
                    corr = epsilon * r * u / norm #http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
                    noisy_img = img + corr
                    inputs_pert[id] = np.ma.clip(noisy_img, 0, 1) #clip values below 0 and over 1
            else:
                print('Incorrect type of noise')
                break
        else:
            inputs_pert = inputs
        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)
        
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

    acc = 100.*correct/total
    return(acc)

def eval_metric(modelfilename, eval_epsilons, noise, train=False):
    test_transforms=transforms.Compose([transforms.ToTensor()#, #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ])
    test_loader_c = torch.utils.data.DataLoader(
        './experiments/data/cifar-10-c',
        batch_size=1024, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./experiments/data', train=False, download=True, transform=test_transforms),
        batch_size=50, shuffle=False)
    # Load model
    model = WideResNet(28, 10, 0.3, 10)
    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []
    for eval_epsilon in eval_epsilons:
        print("Epsilon: ", eval_epsilon)
        #if eval_epsilon == 1:
        #    acc = compute_metric(test_loader_c, model, 0, noise)
        #else:
        acc = compute_metric(test_loader, model, eval_epsilon, noise)
        print(acc)
        accs.append(acc)

    return accs

if __name__ == '__main__':
    # Read command arguments
    parser = argparse.ArgumentParser(description='Computation of TSRM (total statistical robustness metric)')
    parser.add_argument('--model_loc', help='location of saved model')
    parser.add_argument('--prefix', help='prefix for saving results')
    args = parser.parse_args()

    # .experiments/models/cifar_epsilon_{args.epsilon}_run_{args.run}
    modelfilename = os.path.join(args.model_loc, args.prefix+'.pth')
    eval_epsilons = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, epsilon_min, 0.125, 0.15, 0.175, 0.2]
    eval_metric(modelfilename, eval_epsilons)
