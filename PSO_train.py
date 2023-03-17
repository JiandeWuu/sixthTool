import time
import tqdm
import json
import argparse
import multiprocessing

import numpy as np

from sklearn import metrics
from itertools import repeat
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
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
parser.add_argument('-t', '--max_iter', default=1000, type=int, help='hpo max_iter')
parser.add_argument('-seed', '--set_seed', default=1234, type=int, help='seed')

args = parser.parse_args()

print("Input file: %s" % (args.input))
X = np.load(args.input)
print("Label file: %s" % (args.label))
y = np.load(args.label)

size = args.size
fold = args.fold
max_iter = args.max_iter
num_evals = args.num_evals
set_seed = args.set_seed
popu = args.popu

if X.shape[0] != y.shape[0]:
    raise Exception("input file and label file not equal", (X.shape, y.shape))

class multi_ensemble_svm():
    def __init__(self, processes=-1):
        if processes == -1:
            self.processes = multiprocessing.cpu_count()
        else:
            self.processes = processes
            
        self.model_array = None
        self.model_size = None
        self.hp = {
            "kernel": 'C_linear', "C":0, "logGamma":0, "degree":0, "coef0":0, "n":0.5, "max_iter":1e7
        }
    
    def train(self, data, label, ensemble_data_size=1, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=1e7):
        train_time = time.time()
        self.hp["kernel"] = kernel
        self.hp["C"] = C
        self.hp["logGamma"] = logGamma
        self.hp["degree"] = degree
        self.hp["coef0"] = coef0
        self.hp["n"] = n
        self.hp["max_iter"] = max_iter
        
        x, y = svm_function.ensemble_data(data, label, size=ensemble_data_size)
        self.model_size = len(x)
        
        
        pool = multiprocessing.Pool(processes=self.processes)
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        result = pool.starmap(self._svm_train, tqdm.tqdm(zip(x, y), total=self.model_size))
        self.model_array = result
        print("ensemble svm train: %.2fs" % (time.time() - train_time))
        pool.close()
        return None

    def _svm_train(self, d, l):
        arr = np.arange(len(l))
        np.random.shuffle(arr)
        d = d[arr]
        l = l[arr]
        
        m = svm_function.svm_train_model(d, l, self.hp["kernel"], self.hp["C"], self.hp["logGamma"], self.hp["degree"], self.hp["coef0"], self.hp["n"], max_iter=self.hp["max_iter"])
        return m
    
    def test(self, x, y):
        pred_y, pred_y_score = self.predict(x)
            
        return metrics.roc_auc_score(y, pred_y_score)
    
    def _svm_predict(self, m, x):
        return m.predict(x)
    
    def predict(self, x):
        output = None
        pool = multiprocessing.Pool(processes=self.processes)
        output = pool.starmap(self._svm_predict, tqdm.tqdm(zip(self.model_array, repeat(x)), total=self.model_size))
        pred_y_score = np.sum(output, axis=0) / self.model_size
        pred_y = np.where(pred_y_score >= 0.5, 1, 0)
        
        pool.close()
        return pred_y, pred_y_score

def cv_mp_esvm(data_x, data_y, fold=5,
                kernel='C_rbf', 
                C=0, 
                logGamma=0, 
                degree=3, 
                coef0=0, 
                n=0, 
                size=10, 
                max_iter=1000):
    cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, fold)
        
    auroc_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        clf = multi_ensemble_svm()
        clf.train(x_train, y_train, ensemble_data_size=size, kernel=kernel, 
                C=C, 
                logGamma=logGamma, 
                degree=degree, 
                coef0=coef0, 
                n=n, 
                max_iter=max_iter)
        auroc_array.append(clf.test(x_test, y_test))
    return sum(auroc_array) / len(auroc_array)

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
    n = x[6]
    
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
    def __init__(self, X_train, y_train, alpha=0.99, size=1, max_iter=1000, fold=5):
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
        auroc = cv_mp_esvm(self.X_train[:, selected], self.y_train, fold=self.fold, size=self.size, max_iter=self.max_iter,
                    **params 
                    )
        return 1 - auroc
    
problem = eSVMFeatureSelection(X, y, size=size, max_iter=max_iter, fold=fold)

task = Task(problem, max_iters=num_evals)
algorithm = ParticleSwarmOptimization(population_size=popu, seed=set_seed)
best_features, best_fitness = algorithm.run(task)


params = get_params(best_features[:7])
selected_features = best_features[7:] > 0.5

subset_auroc = cv_mp_esvm(X[:, selected_features], y, size=size, max_iter=max_iter, fold=fold, **params)
all_auroc = cv_mp_esvm(X, y, size=size, max_iter=max_iter, fold=fold, **params)

print("params:")
print(params)
print("size: %s, max_iter: %s" % (size, max_iter))
print('Number of selected features:', selected_features.sum())
print("Subset roc_score: %s" % subset_auroc)
print("All roc_score: %s" % all_auroc)

json_dict = {
    "all_auroc": all_auroc,
    "subset_auroc": subset_auroc,
    "selected_features": list(selected_features.astype(str)),
}
print(json_dict)
with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dict, fp)
print("total time: %2.f" % (time.time() - total_time))
