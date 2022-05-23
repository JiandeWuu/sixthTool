import json
import time
import argparse

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from Function import svm_function
from Function.ensemble_svm import ensemble_svm
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output.json', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='Ensemble SVM')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
# parser.add_argument('-p', '--pmap', default=1, type=int, help='hpo pmap')
# parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
parser.add_argument('-hp', '--hyerperparameter', default="", type=str, help='hyerperparameter json file')


parser.add_argument('-k', '--kernel', type=str, help='svm_kernel_type')
parser.add_argument('-d', '--degree', default=3, type=int, help='set degree in kernel function (default 3)')
parser.add_argument('-g', '--logGamma', type=float, help='set logGamma in kernel function')
parser.add_argument('-r', '--coef0', default=0, type=float, help='set coef0 in kernel function (default 0)')
parser.add_argument('-c', '--cost', default=1, type=float, help='set the parameter C (default 1)')
parser.add_argument('-n', '--nu', default=0.5, type=float, help='set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)')

args = parser.parse_args()

print("Input file: %s" % (args.input))
data_x = np.load(args.input)
print("Label file: %s" % (args.label))
data_y = np.load(args.label)
if data_x.shape[0] != data_y.shape[0]:
    raise Exception("input file and label file not equal", (data_x.shape, data_y.shape))

if args.hyerperparameter == "":
    kernel = args.kernel
    degree = args.degree
    logGamma = args.logGamma
    coef0 = args.coef0
    cost = args.cost
    nu = args.nu
else:
    with open(args.hyerperparameter, newline='') as jsonfile:
        hp = json.load(jsonfile)
        kernel = hp["kernel"]
        degree = hp["degree"]
        logGamma = hp["logGamma"]
        coef0 = hp["coef0"]
        C = hp["C"]
        # nu = hp["nu"]
        nu = args.nu

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

def eval_svm_model(data_x, data_y, fold, method, kernel, C, logGamma, degree, coef0, nu):
    cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, fold)

    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    f1sc_array = []
    cm_array = []

    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        
        if method == 'svm':
            model = svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, nu)
        elif method == 'esvm':
            model = esvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, nu)
            
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        cm_array.append(np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]).tolist())
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        acc_array.append((tn + tp) / (tn + fp + fn + tp))
        recall_array.append(tp / (fn + tp))
        prec_array.append(tp / (fp + tp))
        spec_array.append(tn / (tn + fp))
        f1sc_array.append(2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp))))


    json_dcit = {
        "fold Accy": acc_array,
        "avg Accy": sum(acc_array) / len(acc_array),
        "std Accy": np.std(acc_array),
        "fold Recall": recall_array,
        "avg Recall": sum(recall_array) / len(recall_array),
        "std Recall": np.std(recall_array),
        "fold Prec": prec_array,
        "avg Prec": sum(prec_array) / len(prec_array),
        "std Prec": np.std(prec_array),
        "fold Spec": spec_array,
        "avg Spec": sum(spec_array) / len(spec_array),
        "std Spec": np.std(spec_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "std F1sc": np.std(f1sc_array),
        "confusion matrix": cm_array
    }
    
    return json_dcit

json_dcit = eval_svm_model(data_x, data_y, args.fold, args.method, kernel, C, logGamma, degree, coef0, nu)

with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dcit, fp)

print("Output file: %s" % (args.output))
