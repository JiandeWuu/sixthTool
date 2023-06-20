import time
import json
import argparse

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
parser.add_argument('-nor', '--normalize', default=False, type=bool, help='normalize')
args = parser.parse_args()

import numpy as np
import optunity
import optunity.metrics


if args.pmap == 1:
    from sklearnex import patch_sklearn 
    patch_sklearn()
else:
    pmap = optunity.parallel.create_pmap(args.pmap)

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import *
from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict

from Function import svm_function

total_time = time.time()



seed = 1212

esvm_space = {'kernel': {
                    'C_linear': {'C': [-10, 10]},
                    'C_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'C_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'C_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    'Nu_linear': {'n': [1e-2, 1]},
                    'Nu_poly': {'logGamma': [-10, 10], 'n': [1e-2, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'Nu_rbf': {'logGamma': [-10, 10], 'n': [1e-2, 1]},
                    'Nu_sigmoid': {'logGamma': [-10, 10], 'n': [1e-2, 1], 'coef0': [-10, 10]}
                    }
        }

libsvm_space = {'kernel': {
                    '00': {'C': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]},
                    '01': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]},
                    '02': {'logGamma': [-10, 10], 'C': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]},
                    '03': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]},
                    # '10': {'n': [0, 0.9], 'w0': [0, 100], 'w1': [0, 100]},
                    # '11': {'logGamma': [-10, 10], 'n': [0, 0.9], 'degree': [1, 10], 'coef0': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]},
                    # '12': {'logGamma': [-10, 10], 'n': [0, 0.9], 'w0': [0, 100], 'w1': [0, 100]},
                    # '13': {'logGamma': [-10, 10], 'n': [0, 0.9], 'coef0': [-10, 10], 'w0': [0, 100], 'w1': [0, 100]}
                    }
        }

svm_space = {'kernel': {
                    'C_linear': {'C': [-10, 10]},
                    'C_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'C_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'C_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    'Nu_linear': {'n': [1e-2, 1-1e-2]},
                    'Nu_poly': {'logGamma': [-10, 10], 'n': [1e-2, 1-1e-2], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'Nu_rbf': {'logGamma': [-10, 10], 'n': [1e-2, 1-1e-2]},
                    'Nu_sigmoid': {'logGamma': [-10, 10], 'n': [1e-2, 1-1e-2], 'coef0': [-10, 10]}
                    }
        }

print("Input file: %s" % (args.input))
data_x = np.load(args.input)
print("Label file: %s" % (args.label))
data_y = np.load(args.label)

if args.normalize:
    print("Normalize: %s" % (args.normalize))
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)

max_iter = args.max_iter

if data_x.shape[0] != data_y.shape[0]:
    raise Exception("input file and label file not equal", (data_x.shape, data_y.shape))


def esvm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, size=args.size, max_iter=max_iter):
    classifier, kernel = kernel.split("_")
    try:
        model = svm_function.esvm_train_model(x_train, y_train, classifier=classifier, kernel=kernel, C=C, logGamma=logGamma, degree=degree, coef0=coef0, n=n, size=size, max_iter=max_iter, log=True)
        roc_score = model.test(x_test, y_test)
    except:
        roc_score = 0.5
    print("AUROC: %.2f" % roc_score)
    return roc_score

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=max_iter):
    classifier, kernel = kernel.split("_")
    try:
        model = svm_function.svm_train_model(x_train, y_train, classifier, kernel, C, logGamma, degree, coef0, n, max_iter, log=True)
        y_test_proba = model.predict_proba(x_test)
        roc_score = metrics.roc_auc_score(y_test, y_test_proba)
    except:
        print("error return 0.5.")
        roc_score = 0.5
    print("AUROC: %.2f" % (roc_score))
    return roc_score

   
cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=args.fold)

print("Method model: %s" % (args.method))
if args.method == 'svm':
    cv_svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
    if args.pmap == 1:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, svm_space, num_evals=args.num_evals)
    else:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, svm_space, num_evals=args.num_evals, pmap=pmap)
    print("optunity done.")
    
    json_dcit = svm_function.cv_svm_perf(data_x, data_y, 
                                         fold=args.fold, 
                                         kernel=optimal_svm_pars["kernel"], 
                                         C=0 if optimal_svm_pars["kernel"][0] != "C" else optimal_svm_pars["C"], 
                                         logGamma=optimal_svm_pars["logGamma"], 
                                         degree=optimal_svm_pars["degree"], 
                                         coef0=optimal_svm_pars["coef0"], 
                                         n=0.5 if optimal_svm_pars["kernel"][0] == "C" else optimal_svm_pars["n"], 
                                         max_iter=max_iter)
    
elif args.method == 'libsvm':
    cv_libsvm_tuned_auroc = cv_decorator(libsvm_tuned_auroc)
    optimal_svm_pars, info, _ = optunity.maximize_structured(cv_libsvm_tuned_auroc, libsvm_space, num_evals=args.num_evals, pmap=pmap)
    print("optunity done.")
    
    cv_libsvm_perf(data_x, data_y, fold=args.fold, kernel=optimal_svm_pars["kernel"], C=optimal_svm_pars["C"], logGamma=optimal_svm_pars["logGamma"], degree=optimal_svm_pars["degree"], coef0=optimal_svm_pars["coef0"], w0=optimal_svm_pars["w0"], w1=optimal_svm_pars["w1"], n=0.5 if optimal_svm_pars["kernel"][0] == "0" else optimal_svm_pars["n"])

elif args.method == 'esvm':
    cv_esvm_tuned_auroc = cv_decorator(esvm_tuned_auroc)
    if args.pmap == 1:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals)
    else:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals, pmap=pmap)
        
    print("optunity done.")

    json_dcit = svm_function.cv_esvm_perf(data_x, data_y, 
                                          fold=args.fold, 
                                          kernel=optimal_svm_pars["kernel"], 
                                          C=0 if optimal_svm_pars["kernel"][0] != "C" else optimal_svm_pars["C"], 
                                          logGamma=optimal_svm_pars["logGamma"], 
                                          degree=optimal_svm_pars["degree"], 
                                          coef0=optimal_svm_pars["coef0"], 
                                          n=0.5 if optimal_svm_pars["kernel"][0] == "C" else optimal_svm_pars["n"], 
                                          size=args.size, 
                                          max_iter=max_iter)



print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)
print("total time: %2.f" % (time.time() - total_time))

df = optunity.call_log2dataframe(info.call_log)
df = df.sort_values(by=['value'], ascending=False)
df.to_csv(args.output)
with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dcit, fp)