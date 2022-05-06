import time
import json
import argparse

import numpy as np
import pandas as pd
import optunity
import optunity.metrics

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

total_time = time.time()

esvm_space = {'kernel': {
                    '00': {'C': [-10, 10]},
                    '01': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '02': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    '03': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    '10': {'n': [0, 1]},
                    '11': {'logGamma': [-10, 10], 'n': [0, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '12': {'logGamma': [-10, 10], 'n': [0, 1]},
                    '13': {'logGamma': [-10, 10], 'n': [0, 1], 'coef0': [-10, 10]}
                    }
        }

svm_space = {'kernel': {
                    'C_linear': {'C': [-10, 10]},
                    'C_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'C_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'C_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    # 'Nu_linear': {'n': [0, 1]},
                    # 'Nu_poly': {'logGamma': [-10, 10], 'n': [0, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    # 'Nu_rbf': {'logGamma': [-10, 10], 'n': [0, 1]},
                    # 'Nu_sigmoid': {'logGamma': [-10, 10], 'n': [0, 1], 'coef0': [-10, 10]}
                    }
        }

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output.csv', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='Ensemble SVM')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('-p', '--pmap', default=1, type=int, help='hpo pmap')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
args = parser.parse_args()

print("Input file: %s" % (args.input))
data_x = np.load(args.input)
print("Label file: %s" % (args.label))
data_y = np.load(args.label)

if data_x.shape[0] != data_y.shape[0]:
    raise Exception("input file and label file not equal", (data_x.shape, data_y.shape))

pmap = optunity.parallel.create_pmap(args.pmap)

seed = 1212

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

def esvm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='00', C=0, logGamma=0, degree=0, coef0=0, n=0.5):
    model = esvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n)
    roc_score, pred_score = model.test(x_test, y_test)
    return roc_score

def svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    kernel = kernel.split("_")
    if C:
        C = float(C)
    if logGamma:
        logGamma = float(logGamma)
    if degree:
        degree = int(degree)
    if coef0:
        coef0 = float(coef0)
    if n:
        n = float(n)
    
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

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5):
    model = svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n)
    y_pred = model.predict(x_test)
    roc_score = metrics.roc_auc_score(y_test, y_pred)
    return roc_score

def cv_svm_perf(data_x, data_y, fold=5, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5):
    cv_x, cv_y = svm_function.CV(data_x, data_y, fold, seed=1212)
    
    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    f1sc_array = []
    cm_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        model = svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        # np.save("%s_y_fold%s" % (args.output, i), np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]))
        
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
        "fold Recall": recall_array,
        "avg Recall": sum(recall_array) / len(recall_array),
        "fold Prec": prec_array,
        "avg Prec": sum(prec_array) / len(prec_array),
        "fold Spec": spec_array,
        "avg Spec": sum(spec_array) / len(spec_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "confusion matrix": cm_array
    }
    
    with open('%s.json' % (args.output), 'w') as fp:
        json.dump(json_dcit, fp)
        
cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=args.fold)

if args.method == 'svm':
    print("Method model: %s" % (args.method))
    cv_svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
    optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, svm_space, num_evals=args.num_evals, pmap=pmap)
    print("optunity done.")
    
    cv_svm_perf(data_x, data_y, fold=args.fold, kernel=optimal_svm_pars["kernel"], C=optimal_svm_pars["C"], logGamma=optimal_svm_pars["logGamma"], degree=optimal_svm_pars["degree"], coef0=optimal_svm_pars["coef0"], n=0.5)
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=1/args.fold, random_state=1212)
    # model = svm_train_model(x_train, y_train, kernel=optimal_svm_pars["kernel"], C=optimal_svm_pars["C"], logGamma=optimal_svm_pars["logGamma"], degree=optimal_svm_pars["degree"], coef0=optimal_svm_pars["coef0"], n=0.5)
    
    # y_train_pred = model.predict(x_train)
    # np.save(args.output + "_y_train", np.array([y_train, y_train_pred]))
    # y_test_pred = model.predict(x_test)
    # np.save(args.output + "_y_test", np.array([y_test, y_test_pred]))
    
elif args.method == 'esvm':
    print("Method model: %s" % (args.method))
    cv_esvm_tuned_auroc = cv_decorator(esvm_tuned_auroc)
    optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals, pmap=pmap)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=1/args.fold, random_state=1212)
    model = esvm_train_model(x_train, y_train, kernel=optimal_svm_pars["kernel"], C=optimal_svm_pars["C"], logGamma=optimal_svm_pars["logGamma"], degree=optimal_svm_pars["degree"], coef0=optimal_svm_pars["coef0"], n=0.5)
    
    y_train_pred = model.predict(x_train)
    np.save(args.output + "_y_train", np.array([y_train, y_train_pred]))
    y_test_pred = model.predict(x_test)
    np.save(args.output + "_y_test", np.array([y_test, y_test_pred]))


print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)
print("total time: %2.f" % (time.time() - total_time))

df = optunity.call_log2dataframe(info.call_log)
df = df.sort_values(by=['value'], ascending=False)
df.to_csv(args.output)