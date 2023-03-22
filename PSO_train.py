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
proc = args.proc

print("size=%s, fold=%s, max_iter=%s, num_evals=%s, set_seed=%s, popu=%s, proc=%s" % (
    size, fold, max_iter, num_evals, set_seed, popu, proc
))

if X.shape[0] != y.shape[0]:
    raise Exception("input file and label file not equal", (X.shape, y.shape))

class multi_ensemble_svm():
    def __init__(self, processes=-1):
        if processes == -1:
            self.processes = multiprocessing.cpu_count()
        else:
            self.processes = processes
        self.model_array = []
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
        
        self.model_array = []
        
        x, y = svm_function.ensemble_data(data, label, size=ensemble_data_size)
        self.model_size = len(x)
        
        
        pool = multiprocessing.Pool(processes=self.processes)
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(self._svm_train, tqdm.tqdm(zip(x, y), total=self.model_size))
        pool.close()
        self.model_array = results
        print("ensemble svm train: %.2fs" % (time.time() - train_time))
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
    
    def _svm_predict(self, i, x):
        return self.model_array[i].predict(x)
    
    def predict(self, x):
        
        pool = multiprocessing.Pool(processes=self.processes)
        output = pool.starmap(self._svm_predict, tqdm.tqdm(zip(range(self.model_size), itertools.repeat(x)), total=self.model_size))
        # output = []
        # for i in range(self.model_size):
        #     output.append(self.model_array[i].predict(x))
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
        
    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    npv_array = []
    f1sc_array = []
    auroc_array = []
    cm_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        clf = multi_ensemble_svm(processes=proc)
        clf.train(x_train, y_train, ensemble_data_size=size, kernel=kernel, 
                C=C, 
                logGamma=logGamma, 
                degree=degree, 
                coef0=coef0, 
                n=n, 
                max_iter=max_iter)
        y_train_pred, _ = clf.predict(x_train)
        y_test_pred, y_test_pred_score = clf.predict(x_test)
        
        auroc_array.append(metrics.roc_auc_score(y_test, y_test_pred_score))
        cm_array.append(np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]).tolist())
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        acc_array.append((tn + tp) / (tn + fp + fn + tp))
        recall_array.append(tp / (fn + tp))
        prec_array.append(tp / (fp + tp))
        spec_array.append(tn / (tn + fp))
        npv_array.append(tn / (tn + fn))
        f1sc_array.append(2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp))))
    
    
    json_dict = {
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
        "fold Npv": npv_array,
        "avg Npv": sum(npv_array) / len(npv_array),
        "std Npv": np.std(npv_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "std F1sc": np.std(f1sc_array),
        "fold AUROC": auroc_array,
        "avg AUROC": sum(auroc_array) / len(auroc_array),
        "std AUROC": np.std(auroc_array),
        "confusion matrix": cm_array
    }
    
    return json_dict

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
        # auroc = cv_mp_esvm(self.X_train[:, selected], self.y_train, fold=self.fold, size=self.size, max_iter=self.max_iter,
        #             **params 
        #             )['avg AUROC']
        auroc = svm_function.cv_esvm_perf(self.X_train[:, selected], self.y_train, fold=self.fold, size=self.size, max_iter=self.max_iter,
                    **params 
                    )['avg AUROC']
        return 1 - auroc
    
problem = eSVMFeatureSelection(X, y, size=size, max_iter=max_iter, fold=fold)

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
