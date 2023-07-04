import time
import tqdm
import json
import warnings
import argparse
import itertools
import multiprocessing

import numpy as np

# from sklearnex import patch_sklearn 
# patch_sklearn()

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning

from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmOptimization

from Function import svm_function

# Filter out ConvergenceWarning, RuntimeWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
            "classifier": 'SVC', 
            "kernel": 'linear', 
            "C":0, 
            "gamma":0, 
            "degree":0, 
            "coef0":0, 
            "nu":0.5, 
            "max_iter":1e7
        }
    
    def train(self, data, label, ensemble_data_size=1, classifier='SVC', kernel='linear', C=0, gamma=0, degree=0, coef0=0, nu=0.5, max_iter=1e7):
        train_time = time.time()
        self.hp["classifier"] = classifier
        self.hp["kernel"] = kernel
        self.hp["C"] = C
        self.hp["gamma"] = gamma
        self.hp["degree"] = degree
        self.hp["coef0"] = coef0
        self.hp["nu"] = nu
        self.hp["max_iter"] = max_iter
        
        self.model_array = []
        
        x, y = svm_function.ensemble_data(data, label, size=ensemble_data_size)
        self.model_size = len(x)
        
        
        pool = multiprocessing.Pool(processes=self.processes)
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
        
        m = svm_function.svm_train_model(x_train=d, 
                                         y_train=l, 
                                         classifier=self.hp["classifier"], 
                                         kernel=self.hp["kernel"], 
                                         C=self.hp["C"], 
                                         gamma=self.hp["gamma"], 
                                         degree=self.hp["degree"], 
                                         coef0=self.hp["coef0"], 
                                         nu=self.hp["nu"], 
                                         max_iter=self.hp["max_iter"],
                                         log=True)
        return m
    
    def test(self, x, y):
        pred_y, pred_y_score = self.predict(x)
            
        return metrics.roc_auc_score(y, pred_y_score)
    
    def _svm_predict(self, i, x):
        return self.model_array[i].predict(x)
    
    def predict(self, x):
        
        pool = multiprocessing.Pool(processes=self.processes)
        output = pool.starmap(self._svm_predict, tqdm.tqdm(zip(range(self.model_size), itertools.repeat(x)), total=self.model_size))

        pred_y_score = np.sum(output, axis=0) / self.model_size
        pred_y = np.where(pred_y_score >= 0.5, 1, 0)
        
        pool.close()
        return pred_y, pred_y_score

def cv_mp_esvm(data_x, data_y, 
               fold=5,
               classifier='SVC', 
               kernel='rbf', 
               C=0, 
               gamma=0, 
               degree=3, 
               coef0=0, 
               nu=0, 
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
        clf.train(data=x_train, 
                  label=y_train, 
                  ensemble_data_size=size, 
                  classifier=classifier, 
                  kernel=kernel, 
                  C=C, 
                  gamma=gamma, 
                  degree=degree, 
                  coef0=coef0, 
                  nu=nu, 
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
    classifier = "NuSVC" if x[0] > 0.5 else "SVC"
    kernel_list = ['linear',
                    'poly', 
                    'rbf', 
                    'sigmoid']
    kernel = kernel_list[int(x[1] * 3)]
    # kernel = "%s_%s" % (C_Nu, kernel)
    C = (20 * x[2]) - 10
    gamma = (20 * x[3]) - 10
    degree = int(x[4] * 10)
    coef0 = (20 * x[5]) - 10
    nu = x[6] if x[6] != 0 else 1e-7
    
    params = {
        'classifier':classifier, 
        'kernel':kernel, 
        'C':C, 
        'gamma':gamma, 
        'degree':degree, 
        'coef0':coef0, 
        'nu':nu, 
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
        
        # print("avg %.4f, std %.4f" % (sum(x) / len(x), np.std(x)))
        if num_selected == 0:
            print("num_selected=0")
            auroc = 0
        else:
            # auroc = cv_mp_esvm(data_x=self.X_train[:, selected], 
            #                    data_y=self.y_train, 
            #                    fold=self.fold, 
            #                    size=self.size, 
            #                    max_iter=self.max_iter,
            #                    **params 
            #                    )['avg AUROC']
            try:
                auroc = svm_function.cv_esvm_perf(data_x=self.X_train[:, selected], 
                                                data_y=self.y_train, 
                                                fold=self.fold, 
                                                size=self.size, 
                                                max_iter=self.max_iter, 
                                                log=True,
                                                **params 
                                                )['avg AUROC']
            except:
                print("error auroc=0")
                auroc = 0
        print("score: %.4f" % (self.alpha * (1 - auroc) + (1 - self.alpha) * (num_selected / self.X_train.shape[1])))
        return self.alpha * (1 - auroc) + (1 - self.alpha) * (num_selected / self.X_train.shape[1])
    
problem = eSVMFeatureSelection(X, y, size=size, max_iter=max_iter, fold=fold)

task = Task(problem=problem, max_iters=num_evals)
algorithm = ParticleSwarmOptimization(population_size=popu, seed=set_seed)
best_features, best_fitness = algorithm.run(task)


model_params = get_params(best_features[:7])
model_params['size'] = size
model_params['fold'] = fold
model_params['max_iter'] = max_iter
model_params['log'] = True

selected_features = best_features[7:] > 0.5

print("subset eSVM train:")
subset_perf = svm_function.cv_esvm_perf(X[:, selected_features], y, **model_params)


print("model_params:")
print(model_params)
print('Number of selected features: %s / %s' % (selected_features.sum(), X.shape[1]))
print("Subset roc_score: %s" % subset_perf['avg AUROC'])

parmas = {}
parmas["input"] = args.input
parmas["label"] = args.label
parmas["num_evals"] = args.num_evals
parmas["popu"] = args.popu
parmas["set_seed"] = args.set_seed

total_time = time.time() - total_time
json_dict = {
    # "all_perf": all_perf,
    "total_time": total_time,
    "subset_perf": subset_perf,
    "model_params": model_params,
    "params": parmas,
    "selected_features": list(selected_features.astype(str)),
}

with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dict, fp)
print("total time: %2.f" % (total_time))
