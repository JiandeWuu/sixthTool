import time
import tqdm
import json
import argparse
import itertools
import multiprocessing

import numpy as np

from sklearnex import patch_sklearn 
patch_sklearn()

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization

from Function import svm_function

total_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output.csv', type=str, help='output file')
# parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-p', '--proc', default=-1, type=int, help='multiprocessing')
parser.add_argument('-u', '--popu', default=25, type=int, help='population size')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
parser.add_argument('-t', '--max_iter', default=1000, type=int, help='hpo max_iter')
parser.add_argument('-seed', '--set_seed', default=1234, type=int, help='seed')

args = parser.parse_args()

print("Input file: %s" % (args.input))
X = np.load(args.input)
print("Label file: %s" % (args.label))
y = np.load(args.label)

fold = args.fold
max_iter = args.max_iter
num_evals = args.num_evals
set_seed = args.set_seed
popu = args.popu
proc = args.proc

print("fold=%s, max_iter=%s, num_evals=%s, set_seed=%s, popu=%s, proc=%s" % (
    fold, max_iter, num_evals, set_seed, popu, proc
))

if X.shape[0] != y.shape[0]:
    raise Exception("input file and label file not equal", (X.shape, y.shape))


def get_params(x):
    C_Nu = "Nu" if x[0] > 0.5 else "C"
    kernel_list = ['linear',
                    'poly', 
                    'rbf', 
                    'sigmoid']
    kernel = kernel_list[int(x[1] * 3)]
    kernel = "%s_%s" % (C_Nu, kernel)
    C = (20 * x[2]) - 10
    logGamma = (20 * x[3]) - 10
    degree = int(x[4] * 10)
    coef0 = (20 * x[5]) - 10
    n = x[6] if x[6] != 0 else 1e-7
    
    params = {
        'kernel':kernel, 
        'C':C, 
        'logGamma':logGamma, 
        'degree':degree, 
        'coef0':coef0, 
        'n':n, 
    }
    return params

class eSVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99, max_iter=1000, fold=5):
        super().__init__(dimension=X_train.shape[1] + 7, lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.size = size
        self.max_iter = max_iter
        self.fold = fold
        
    def _evaluate(self, x):
        params = get_params(x[:7])
        
        x = x[7:]
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        # auroc = cv_mp_esvm(self.X_train[:, selected], self.y_train, fold=self.fold, size=self.size, max_iter=self.max_iter,
        #             **params 
        #             )['avg AUROC']
        auroc = svm_function.cv_esvm_perf(self.X_train[:, selected], self.y_train, fold=self.fold, size=self.size, max_iter=self.max_iter,
                    **params 
                    )['avg AUROC']
        return 1 - auroc
    
problem = eSVMFeatureSelection(X, y, max_iter=max_iter, fold=fold)

task = Task(problem, max_iters=num_evals)
algorithm = ParticleSwarmOptimization(population_size=popu, seed=set_seed)
best_features, best_fitness = algorithm.run(task)


params = get_params(best_features[:7])
params['size'] = size
params['fold'] = fold
params['max_iter'] = max_iter

selected_features = best_features[7:] > 0.5

print("subset eSVM train:")
# subset_perf = cv_mp_esvm(X[:, selected_features], y, **params)
subset_perf = svm_function.cv_esvm_perf(X[:, selected_features], y, **params)

# print("All eSVM train:")
# all_perf = cv_mp_esvm(X, y, **params)
# all_perf = svm_function.cv_esvm_perf(X, y, **params)

print("params:")
print(params)
print('Number of selected features:', selected_features.sum())
print("Subset roc_score: %s" % subset_perf['avg AUROC'])
# print("All roc_score: %s" % all_perf['avg AUROC'])

total_time = time.time() - total_time
json_dict = {
    # "all_perf": all_perf,
    "total_time": total_time,
    "subset_perf": subset_perf,
    "params": params,
    "selected_features": list(selected_features.astype(str)),
}
# print(json_dict)
with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dict, fp)
print("total time: %2.f" % (total_time))
