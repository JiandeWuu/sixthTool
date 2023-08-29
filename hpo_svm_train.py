import time
import json
import warnings
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
parser.add_argument('-perf', '--performance_value', default="auroc", type=str, help='hpo evals performance value, default=auroc, [acc, recall, prec, spec, npv, f1sc]')
parser.add_argument('-timeout', '--timeout', default=None, type=int, help='timeout sec')
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
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from wrapt_timeout_decorator import timeout

from Function import svm_function

total_time = time.time()

# Filter out ConvergenceWarning, RuntimeWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

seed = 1212

esvm_space = {'kernel': {
                    'SVC_linear': {'C': [-10, 10]},
                    'SVC_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'SVC_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'SVC_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    'NuSVC_linear': {'n': [1e-2, 1]},
                    'NuSVC_poly': {'logGamma': [-10, 10], 'n': [1e-2, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'NuSVC_rbf': {'logGamma': [-10, 10], 'n': [1e-2, 1]},
                    'NuSVC_sigmoid': {'logGamma': [-10, 10], 'n': [1e-2, 1], 'coef0': [-10, 10]}
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
                    'SVC_linear': {'C': [-10, 10]},
                    'SVC_poly': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'SVC_rbf': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    'SVC_sigmoid': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    'NuSVC_linear': {'n': [1e-2, 1-1e-2]},
                    'NuSVC_poly': {'logGamma': [-10, 10], 'n': [1e-1, 1-1e-1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    'NuSVC_rbf': {'logGamma': [-10, 10], 'n': [1e-1, 1-1e-1]},
                    'NuSVC_sigmoid': {'logGamma': [-10, 10], 'n': [1e-1, 1-1e-1], 'coef0': [-10, 10]}
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

if args.timeout:
    timeout_ = timeout(args.timeout, use_signals=False)
    svm_train_model = timeout_(svm_function.svm_train_model)
    esvm_train_model = timeout_(svm_function.esvm_train_model)
    timeout_ = timeout(args.timeout * args.fold, use_signals=False)
    cv_svm_perf = timeout_(svm_function.cv_svm_perf)
    cv_esvm_perf = timeout_(svm_function.cv_esvm_perf)
else:
    svm_train_model = svm_function.svm_train_model
    cv_svm_perf = svm_function.cv_svm_perf
    esvm_train_model = svm_function.esvm_train_model
    cv_esvm_perf = svm_function.cv_esvm_perf
    
def get_performance_value(y, y_pred, y_pred_proba, perf=args.performance_value):
    
    if perf == "auroc":
        y_pred_proba = np.where(np.isnan(y_pred_proba), 0, y_pred_proba) 
        y_pred_proba = np.where(np.isfinite(y_pred_proba), y_pred_proba, 0) 
        perf_val = metrics.roc_auc_score(y, y_pred_proba)
    else:
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        if perf == "acc":
            perf_val = (tn + tp) / (tn + fp + fn + tp)
        elif perf == "recall" :
            perf_val = tp / (fn + tp)
        elif perf == "prec" :
            perf_val = tp / (fp + tp)
        elif perf == "spec" :
            perf_val = tn / (tn + fp)
        elif perf == "npv" :
            perf_val = tn / (tn + fn)
        elif perf == "f1sc" :
            perf_val = 2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp)))
        else:
            raise Exception("get_performance_value perf=%s is not in [acc, recall, prec, spec, npv, f1sc, auroc]" % perf)
    return perf_val

def esvm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, size=args.size, max_iter=max_iter):
    classifier, kernel = kernel.split("_")
    try:
        model = esvm_train_model(x_train, y_train, classifier=classifier, kernel=kernel, C=C, gamma=logGamma, degree=degree, coef0=coef0, nu=n, size=size, max_iter=max_iter, log=True)
        y_pred, y_pred_score = model.predict(x_test)
        pref_val = get_performance_value(y=y_test, y_pred=y_pred, y_pred_proba=y_pred_score)
    except:
        print("error return 0.")
        pref_val = 0
    print("%s: %.2f" % (args.performance_value, pref_val))
    return pref_val

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=max_iter):
    start_time = time.time()
    try:
        classifier, kernel = kernel.split("_")
        model = svm_train_model(x_train, y_train, classifier, kernel, C, logGamma, degree, coef0, n, max_iter, log=True)
        y_pred = model.predict(x_test)
        decision_values = model.decision_function(x_test)
        pref_val = get_performance_value(y=y_test, y_pred=y_pred, y_pred_proba=decision_values)
    except Exception as e:
        print("error return 0.", e)
        pref_val = 0
    print("%s: %.2f | %.2fs" % (args.performance_value, pref_val, time.time() - start_time))
    return pref_val

   
cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=args.fold)

print("Method model: %s" % (args.method))
if args.method == 'svm':
    cv_svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
    if args.pmap == 1:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, svm_space, num_evals=args.num_evals)
    else:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, svm_space, num_evals=args.num_evals, pmap=pmap)
    print("optunity done.")
    
    json_dcit = cv_svm_perf(data_x, data_y, 
                                         fold=args.fold, 
                                         classifier=optimal_svm_pars["kernel"].split("_")[0], 
                                         kernel=optimal_svm_pars["kernel"].split("_")[1], 
                                         C=0 if optimal_svm_pars["kernel"].split("_")[0] != "SVC" else optimal_svm_pars["C"], 
                                         gamma=optimal_svm_pars["logGamma"], 
                                         degree=optimal_svm_pars["degree"], 
                                         coef0=optimal_svm_pars["coef0"], 
                                         nu=0.5 if optimal_svm_pars["kernel"].split("_")[0] == "SVC" else optimal_svm_pars["n"], 
                                         max_iter=max_iter,
                                         log=True)
elif args.method == 'esvm':
    cv_esvm_tuned_auroc = cv_decorator(esvm_tuned_auroc)
    if args.pmap == 1:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals)
    else:
        optimal_svm_pars, info, _ = optunity.maximize_structured(cv_esvm_tuned_auroc, esvm_space, num_evals=args.num_evals, pmap=pmap)
        
    print("optunity done.")

    print("Optimal parameters " + str(optimal_svm_pars))
    
    json_dcit = cv_esvm_perf(data_x, data_y, 
                                          fold=args.fold, 
                                          classifier=optimal_svm_pars["kernel"].split("_")[0], 
                                          kernel=optimal_svm_pars["kernel"].split("_")[1], 
                                          C=0 if optimal_svm_pars["kernel"].split("_")[0] != "SVC" else optimal_svm_pars["C"], 
                                          gamma=optimal_svm_pars["logGamma"], 
                                          degree=optimal_svm_pars["degree"], 
                                          coef0=optimal_svm_pars["coef0"], 
                                          nu=0.5 if optimal_svm_pars["kernel"].split("_")[0] == "SVC" else optimal_svm_pars["n"], 
                                          size=args.size, 
                                          max_iter=max_iter,
                                          log=True)
    json_dcit["size"] = args.size

# parameter
json_dcit["input"] = args.input
json_dcit["label"] = args.label
json_dcit["method"] = args.method
json_dcit["normalize"] = args.normalize
json_dcit["max_iter"] = args.max_iter
json_dcit["fold"] = args.fold
json_dcit["performance_value"] = args.performance_value
json_dcit["num_evals"] = args.num_evals
json_dcit["pmap"] = args.pmap
json_dcit["timeout"] = args.timeout

print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)
print("total time: %2.f" % (time.time() - total_time))

df = optunity.call_log2dataframe(info.call_log)
df = df.sort_values(by=['value'], ascending=False)
df.to_csv(args.output)
with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dcit, fp)