import time
import json
import argparse

import numpy as np
import pandas as pd
import optunity
import optunity.metrics

from sklearnex import patch_sklearn 
patch_sklearn()

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

total_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output.csv', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('-p', '--pmap', default=1, type=int, help='hpo pmap')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
parser.add_argument('-t', '--max_iter', default=1000, type=int, help='hpo max_iter')
args = parser.parse_args()

seed = 1212

esvm_space = {'kernel': {
                    'C_linear': {'C': [-10, 10]},
                    'C_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'C_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'C_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    'Nu_linear': {'n': [0, 1]},
                    'Nu_poly': {'logGamma': [-10, 10], 'n': [1e-7, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'Nu_rbf': {'logGamma': [-10, 10], 'n': [1e-7, 1]},
                    'Nu_sigmoid': {'logGamma': [-10, 10], 'n': [1e-7, 1], 'coef0': [-10, 10]}
                    }
        }

print("Input file: %s" % (args.input))
data_x = np.load(args.input)
print("Label file: %s" % (args.label))
data_y = np.load(args.label)

max_iter = args.max_iter

if data_x.shape[0] != data_y.shape[0]:
    raise Exception("input file and label file not equal", (data_x.shape, data_y.shape))

pmap = optunity.parallel.create_pmap(args.pmap)

def esvm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, size=args.size, max_iter=max_iter, **params):
    model = svm_function.esvm_train_model(x_train, y_train, kernel=kernel, C=C, logGamma=logGamma, degree=degree, coef0=coef0, n=n, size=size, max_iter=max_iter)
    roc_score = model.test(x_test, y_test)
    print("AUROC: %.2f" % roc_score)
    return roc_score

for i in range(data_x.shape[1]):
    esvm_space

cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=args.fold)

cv_esvm_tuned_auroc = cv_decorator(esvm_tuned_auroc)
optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals, pmap=pmap)
print("optunity done.")

# json_dcit = svm_function.cv_esvm_perf(data_x, data_y, fold=args.fold, kernel=optimal_svm_pars["kernel"], C=optimal_svm_pars["C"], logGamma=optimal_svm_pars["logGamma"], degree=optimal_svm_pars["degree"], coef0=optimal_svm_pars["coef0"], n=0.5, size=args.size, max_iter=max_iter)
