from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os
os.chdir('C:\\Users\\Admin\\Desktop\\Python\\MSSR')
import numpy as np
import matplotlib.pyplot as plt
from experiments.eval import eval_metric

#from experiments.distance import get_nearest_oppo_dist
#calculate minimal distance of points from different classes
#dist = np.inf #1, 2, np.inf
#traintrain_ret, traintest_ret, testtest_ret = get_nearest_oppo_dist(dist)
#ret = np.array([[traintrain_ret.min(), traintest_ret.min(), testtest_ret.min()],
#       [traintrain_ret.mean(), traintest_ret.mean(), testtest_ret.mean()]])
#df_ret = pd.DataFrame(ret, columns=['Train-Train', 'Train-Test', 'Test-Test'], index=['Minimal Distance', 'Mean Distance'])
#print(df_ret)
#epsilon_min = ret[0, :].min()/2
#print("Epsilon: ", epsilon_min)

#select max.-distance of random perturbations for model training and evaluation
epsilon_min = 0.10588236153125763
#model_epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, epsilon_min, 0.15] #CIFAR-10 epsilon_min value is 0.10588236153125763
#eval_epsilons = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, epsilon_min, 0.125, 0.15, 0.175, 0.2]
model_epsilons = [0.0, 0.005, 0.01, 0.015, 0.02] #CIFAR-10 epsilon_min value is 0.10588236153125763
eval_epsilons = [0.0, 0.005]
model_epsilons_str = ', '.join(map(str, model_epsilons))
eval_epsilons_str = ', '.join(map(str, eval_epsilons))

runs = 16

# Train network on CIFAR-10 for natural training, two versions of corruption training, and PGD adversarial training.
# Progressively smaller learning rates are used over training
print('Beginning training of Wide ResNet networks on CIFAR-10')
for run in range(0, 16):
    print("Training run #", run)
    for train_epsilon in model_epsilons:
        print("Corruption training epsilon: ", train_epsilon)
        cmd0 = 'python experiments/train.py --epsilon={} --epochs=20 --lr=0.01 --run={}'.format(
            train_epsilon, run)
        cmd1 = 'python experiments/train.py --resume --epsilon={} --epochs=5 --lr=0.002 --run={}'.format(
            train_epsilon, run)
        cmd2 = 'python experiments/train.py --resume --epsilon={} --epochs=5 --lr=0.0004 --run={}'.format(
            train_epsilon, run)
        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)

# Calculate metrics (A-TSRM and adv. accuracy), evaluating each trained network on each metric
print('Beginning metric evaluation')
# Evaluation on train/test set respectively
all_test_metrics = np.empty([len(eval_epsilons), len(model_epsilons), runs])
all_mscr = np.empty([len(model_epsilons), runs])
std_mscr = np.empty(len(model_epsilons))
all_dif = np.empty([4, runs])
all_avg = np.empty([4])
all_std = np.empty([4])
avg_test_metrics = np.empty([len(eval_epsilons), len(model_epsilons)])
std_test_metrics = np.empty([len(eval_epsilons), len(model_epsilons)])
max_test_metrics = np.empty([len(eval_epsilons), len(model_epsilons)])

#acc1 = np.empty(runs)
#acc_series1 = np.empty(runs)

for run in range(runs):
    print("Metric evaluation for training run #", run)
    test_metrics = np.empty([len(eval_epsilons), len(model_epsilons)])

    # Corruption training, A-TRSM evaluation
    for idx, train_epsilon in enumerate(model_epsilons):
        print("Corruption training epsilon: ", train_epsilon, ", Evaluate on A-TSRM epsilons: ", eval_epsilons_str)
        filename = './experiments/models/cifar_epsilon_{}_run_{}.pth'.format(train_epsilon, run)
        test_metric_col = eval_metric(filename, eval_epsilons, adv=False, train=False)
        test_metrics[:len(eval_epsilons), idx] = np.array(test_metric_col)
        #mscr: calculated from clean and r-separated testing sets
        #all_mscr[idx, run] = (np.array(test_metric_col)[13] - np.array(test_metric_col)[0]) / np.array(test_metric_col)[0]

    all_test_metrics[:len(eval_epsilons), :len(model_epsilons), run] = test_metrics
    all_dif[0, run] = all_test_metrics[0, 1, run] - all_test_metrics[0, 0, run]
    all_dif[1, run] = all_test_metrics[0, 3, run] - all_test_metrics[0, 0, run]
    all_dif[2, run] = all_test_metrics[1, 1, run] - all_test_metrics[1, 0, run]
    all_dif[3, run] = all_test_metrics[1, 3, run] - all_test_metrics[1, 0, run]

    #    acc1[run] = test_metrics[0, 0]
#    acc_series1[run] = acc1[:run+1].mean()

    np.savetxt(
        './results/cifar10_metrics_test_run_{}.csv'.format(
        run), test_metrics, fmt='%1.3f', delimiter=';', header='Networks trained with'
        ' corruptions (epsilon = {}) along columns THEN evaluated on test set using A-TRSM (epsilon = {}) '
        ' along rows'.format(model_epsilons_str, eval_epsilons_str, ))

for idm, model_epsilon in enumerate(model_epsilons):
    #std_mscr[idm] = all_mscr[idm, :].std()
    for ide, eval_epsilon in enumerate(eval_epsilons):
        avg_test_metrics[ide, idm] = all_test_metrics[ide, idm, :runs].mean()
        std_test_metrics[ide, idm] = all_test_metrics[ide, idm, :runs].std()
        max_test_metrics[ide, idm] = all_test_metrics[ide, idm, :runs].max()

all_std[0] = all_dif[0,:].std()
all_avg[0] = all_dif[0,:].mean()
all_std[1] = all_dif[1,:].std()
all_avg[1] = all_dif[1,:].mean()
all_std[2] = all_dif[2,:].std()
all_avg[2] = all_dif[2,:].mean()
all_std[3] = all_dif[3,:].std()
all_avg[3] = all_dif[3,:].mean()
print(all_avg)
print(all_std)
#print(acc1)
#print(acc1.std())
#x1 = list(range(1, runs + 1))
#y1 = acc_series1
#plt.scatter(x1, y1)
#plt.xlabel("Runs")
#plt.ylabel("Average Test Acc")
#plt.title("Convergence of average Test Accuracy over runs")
#plt.legend(["train&test e = 0"], loc='right')
#plt.show()

np.savetxt(
    './results/cifar10_metrics_test_avg.csv',
    avg_test_metrics, fmt='%1.3f', delimiter=';', header='Networks trained with'
    ' corruptions (epsilon = {}) along columns THEN evaluated on test set using A-TRSM (epsilon = {}) '
    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
np.savetxt(
    './results/cifar10_metrics_test_max.csv',
    max_test_metrics, fmt='%1.3f', delimiter=';', header='Networks trained with'
    ' corruptions (epsilon = {}) along columns THEN evaluated on test set using A-TRSM (epsilon = {}) '
    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
np.savetxt(
    './results/diff.csv',
    all_dif, fmt='%1.3f', delimiter=';')
np.savetxt(
    './results/cifar10_metrics_test_std.csv',
    std_test_metrics, fmt='%1.4f', delimiter=';', header='Networks trained with'
    ' corruptions (epsilon = {}) along columns THEN evaluated on test set using A-TRSM (epsilon = {}) '
    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
np.savetxt(
    './results/cifar10_mscr_std.csv',
    std_mscr, fmt='%1.4f', delimiter=';', header='Networks trained with'
    ' corruptions (epsilon = {}) along columns'.format(model_epsilons_str))
