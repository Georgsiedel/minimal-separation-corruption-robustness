from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# calculate r-separation distance of dataset
def get_nearest_oppo_dist(norm):
    
    transform_train = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./experiments/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='./experiments/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    #this function reshapes the dataset dimensions so all input parameters of the datapoints are a list for every point, not multi-dimensional
    for helper_id, (inputs, targets) in enumerate(trainloader):
        train_x = inputs
        train_y = targets
        if len(train_x.shape) > 2:
            train_x = train_x.reshape(len(train_x), -1)

    for helper_id, (inputs, targets) in enumerate(testloader):
        test_x = inputs
        test_y = targets
        if len(test_x.shape) > 2:
            test_x = test_x.reshape(len(test_x), -1)
            
    print("Data loading done for distance evaluation")
    #Fits a One-nearest-neighbour-model for the train dataset and the test dataset, where model data is only the data points that do not feature label "yi"
    def helper_train(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=norm, n_jobs=-1).fit(train_x[train_y != yi])
    nns_train = Parallel(n_jobs=10)(delayed(helper_train)(yi) for yi in np.unique(train_y))

    def helper_test(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=norm, n_jobs=-1).fit(test_x[test_y != yi])
    nns_test = Parallel(n_jobs=10)(delayed(helper_test)(yi) for yi in np.unique(test_y))
    
    #evaluate minimal distances of training data points to training data points of different classes (traintrain_ret),
    #training data points to test data points of different classes (traintest_ret) and
    #test data points to test data points of different classes (testtest_ret),
    traintrain_ret = np.zeros(len(train_x))
    traintest_ret = np.zeros(len(test_x))
    testtest_ret = np.zeros(len(test_x))
    time0 = time.perf_counter()
    for yi in np.unique(train_y):
        dist, _ = nns_train[yi].kneighbors(train_x[train_y == yi], n_neighbors=1)
        traintrain_ret[np.where(train_y == yi)[0]] = dist[:, 0]
        if yi == 0:
            time1 = time.perf_counter()
            print("First Train Train done after ", (time1 - time0), "seconds. Distance calculation time estimation: ", (time1 - time0)*12.5, "seconds.")
    for yi in np.unique(train_y):
        dist, _ = nns_train[yi].kneighbors(test_x[test_y == yi], n_neighbors=1)
        traintest_ret[np.where(test_y == yi)[0]] = dist[:, 0]

    for yi in np.unique(test_y):
        dist, _ = nns_test[yi].kneighbors(test_x[test_y == yi], n_neighbors=1)
        testtest_ret[np.where(test_y == yi)[0]] = dist[:, 0]
    time2 = time.perf_counter()
    print(time2 - time0)

    return traintrain_ret, traintest_ret, testtest_ret


dist = np.inf #1, 2, np.inf
traintrain_ret, traintest_ret, testtest_ret = get_nearest_oppo_dist(dist)
traintrain = np.sort(traintrain_ret)
np.savetxt('./results/traintrain-separation.csv', traintrain, fmt='%1.4f', delimiter=';')

#Visualize a histogram of the minimal distances to a different class for all points
traintest = np.sort(traintest_ret)
testtest = np.sort(testtest_ret)
y1 = np.arange(len(traintrain_ret)) / len(traintrain_ret)
y2 = np.arange(len(traintest_ret)) / len(traintest_ret)
y3 = np.arange(len(testtest_ret)) / len(testtest_ret)

n_bins = 500
fig = plt.figure(figsize=[5,3.5], dpi=300)
fig, axs = plt.subplots(3, 2, sharex='col', tight_layout=True)
axs[0, 0].hist(traintrain_ret, bins=n_bins)
axs[1, 0].hist(traintest_ret, bins=n_bins)
axs[2, 0].hist(testtest_ret, bins=n_bins)
axs[0, 1].plot(traintrain, y1)
axs[1, 1].plot(traintest, y2)
axs[2, 1].plot(testtest, y3)
plt.xlim(0, 1)
axs[0,0].set_title("Train-Train Separation Distribution")
axs[1,0].set_title("Train-Test Separation Distribution")
axs[2,0].set_title("Test-Test Separation Distribution")
axs[0,1].set_title("Train-Train Separation CDF")
axs[1,1].set_title("Train-Test Separation CDF")
axs[2,1].set_title("Test-Test Separation CDF")
fig.savefig(r"results/r-distance-distribution.svg", dpi=300)

fig2 = plt.figure(figsize=[5,3.5], dpi=300)
fig2, axs2 = plt.subplots(1, 1, sharex='col', tight_layout=True)
axs2.hist(traintrain_ret, bins=n_bins)
#axs2[1].plot(traintrain, y1)
plt.xlim(0.2, 0.7)
plt.xlabel("Distance (Linf)")
plt.ylabel("Frequency of points")
axs2.set_title("Train-Train Separation Distribution")
#axs2[1].set_title("Train-Train Separation CDF")
fig2.savefig(r"results/r-distance-distribution2.svg", dpi=300)
