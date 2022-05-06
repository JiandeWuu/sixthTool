import time
import argparse

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import metrics

from Function import svm_function
from Function.ensemble_svm import ensemble_svm
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output.csv', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='Ensemble SVM')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
# parser.add_argument('-p', '--pmap', default=1, type=int, help='hpo pmap')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')

parser.add_argument('-k', '--kernel', type=str, help='svm_kernel_type')
parser.add_argument('-d', '--degree', default=3, type=int, help='set degree in kernel function (default 3)')
parser.add_argument('-g', '--logGamma', type=float, help='set logGamma in kernel function')
parser.add_argument('-r', '--coef0', default=0, type=float, help='set coef0 in kernel function (default 0)')
parser.add_argument('-c', '--cost', default=1, type=float, help='set the parameter C (default 1)')
parser.add_argument('-n', '--nu', type=float, help='set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)')

args = parser.parse_args()

print("Input file: %s" % (args.input))
data_x = np.load(args.input)
print("Label file: %s" % (args.label))
data_y = np.load(args.label)
if data_x.shape[0] != data_y.shape[0]:
    raise Exception("input file and label file not equal", (data_x.shape, data_y.shape))

def esvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    x_train, y_train = svm_function.ensemble_data(x_train, y_train, size=args.size)
    esvm = ensemble_svm()
    
    parameter = "-s " + kernel[0] + " -t " + kernel[1]
    if kernel[0] == '0':
        parameter += " -c " + str(2 ** C)
    else:
        parameter += " -n " + str(n)
        
    if kernel[1] != '0':
        parameter += " -g " + str(2 ** logGamma)
    if kernel[1] == '1':
        parameter += " -d " + str(int(degree))
    if kernel[1] == '1' or kernel[1] == '3':
        parameter += " -r " + str(2 ** coef0)
    esvm.train(x_train, y_train, parameter=parameter)
    
    return esvm

def svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    kernel = kernel.split("_")
    
    if kernel[0] == "C":
        if kernel[1] == "linear":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "poly":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), degree=degree, coef0=(2 ** coef0), class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "rbf":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "sigmoid":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), coef0=(2 ** coef0), class_weight='balanced').fit(x_train, y_train)
    elif kernel[0] == "Nu":
        if kernel[1] == "linear":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "poly":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), degree=degree, coef0=(2 ** coef0), class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "rbf":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), class_weight='balanced').fit(x_train, y_train)
        elif kernel[1] == "sigmoid":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), coef0=(2 ** coef0), class_weight='balanced').fit(x_train, y_train)
    
    return clf

cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, args.fold)

roc_score_array = []
for i in range(args.fold):
    x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
    if args.method == 'svm':
        print("Method model: %s" % (args.method))
        clf = svm_train_model(x_train, y_train, args.kernel, args.C, args.logGamma, args.degree, args.coef0, args.n)
        y_pred = clf.predict(x_test)
        roc_score = metrics.roc_auc_score(y_test, y_pred)
    elif args.method == 'esvm':
        print("Method model: %s" % (args.method))
        
        clf = esvm_train_model(x_train, y_train, args.kernel, args.C, args.logGamma, args.degree, args.coef0, args.n)

auroc = sum(roc_score_array) / len(roc_score_array)
print("file name=%s, time=%.2f" % (file_name, time.time() - file_time))
