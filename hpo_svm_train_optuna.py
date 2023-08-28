import time
import json
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-storage', '--storage', default=None, type=str, help='storage file .log')
parser.add_argument('-n', '--study_name', default=None, type=str, help='study name')
parser.add_argument('-sampler', '--study_sampler', default=None, type=str, help='study sampler, default=None, [RandomSampler, CmaEsSampler]')
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-f', '--fold', default=5, type=int, help='k-fold cross-validation')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('-j', '--n_jobs', default=1, type=int, help='hpo n_jobs')
parser.add_argument('-e', '--num_evals', default=10, type=int, help='hpo num_evals')
parser.add_argument('-t', '--max_iter', default=1000, type=int, help='hpo max_iter')
parser.add_argument('-nor', '--normalize', default=False, type=bool, help='normalize')
parser.add_argument('-perf', '--performance_value', default="AUROC", type=str, help='hpo evals performance value, default=AUROC, [Accy, Recall, Prec, Spec, Npv, F1sc]')
parser.add_argument('-fs', '--feature_selection', default=False, type=bool, help='feature_selection, default=False')
parser.add_argument('-timeout', '--timeout', default=None, type=float, help='timeout sec')
args = parser.parse_args()

if args.n_jobs == 1:
    from sklearnex import patch_sklearn 
    patch_sklearn()

import optuna
import numpy as np

from sklearn.preprocessing import *
from sklearn.exceptions import ConvergenceWarning

from optuna.samplers import CmaEsSampler
from optuna.samplers import RandomSampler


from Function import svm_function

total_time = time.time()

# Filter out ConvergenceWarning, RuntimeWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

optuna.logging.set_verbosity(optuna.logging.WARN)

hp_space = {
    'classifier': ["SVC", "NuSVC"],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'C': {"low": 1e-10, "high": 1e10, "log": True},
    'gamma': {"low": 1e-10, "high": 1e10, "log": True},
    'coef0': {"low": 1e-10, "high": 1e10, "log": True},
    'degree': {"low": 1, "high": 10, "log": False},
    'nu': {"low": 1e-2, "high": 1-1e-2, "log": False},
    'max_iter': args.max_iter
}


if not args.method in ["esvm", "svm"]:
    raise Exception("method is not [esvm, svm]", (args.method))


print("method=%s, max_iter=%s, size=%s, fold=%s, n_jobs=%s, num_evals=%s, performance_value=%s, timeout=%s" % 
      (args.method, args.max_iter, args.size, args.fold, args.n_jobs, args.num_evals, args.performance_value, args.timeout))

print("Input file: %s" % (args.input))
x = np.load(args.input)
print("Label file: %s" % (args.label))
y = np.load(args.label)

if x.shape[0] != y.shape[0]:
    raise Exception("input file and label file not equal", (x.shape, y.shape))

if args.normalize:
    print("Normalize: %s" % (args.normalize))
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

study_sampler = args.study_sampler
if args.study_sampler == "RandomSampler":
    study_sampler = RandomSampler()
if args.study_sampler == "CmaEsSampler":
    study_sampler = CmaEsSampler()

class Objective:

    def __init__(self, x, y, hp_space, feature_selection):
        self.best_perf = None
        self._perf = None
        self.x = x
        self.y = y
        self.hp_space = hp_space
        self.feature_selection = feature_selection
        
    def __call__(self, trial):
        x = self.x.copy()
        if self.feature_selection:
            feature_selected = []
            for i in range(x.shape[1]):
                feature_selected.append(trial.suggest_categorical("feature_%s" % i, [True, False]))
            x = x[:, feature_selected]
        
        classifier_name = trial.suggest_categorical("classifier", self.hp_space["classifier"])
        kernel = trial.suggest_categorical("kernel", self.hp_space["kernel"])
        
        C = None
        nu = None
        gamma = None
        coef0 = None
        degree = None
        
        if classifier_name == "SVC":
            C = trial.suggest_float("C", self.hp_space["C"]["low"], self.hp_space["C"]["high"], log=self.hp_space["C"]["log"])
        
        if classifier_name == "NuSVC":
            nu = trial.suggest_float("nu", self.hp_space["nu"]["low"], self.hp_space["nu"]["high"], log=self.hp_space["nu"]["log"])
        
        if kernel != "linear":
            gamma = trial.suggest_float("gamma", self.hp_space["gamma"]["low"], self.hp_space["gamma"]["high"], log=self.hp_space["gamma"]["log"])
            if kernel != "rbf":
                coef0 = trial.suggest_float("coef0", self.hp_space["coef0"]["low"], self.hp_space["coef0"]["high"], log=self.hp_space["coef0"]["log"])
                if kernel != "sigmoid":
                    degree = trial.suggest_int("degree", self.hp_space["degree"]["low"], self.hp_space["degree"]["high"], log=self.hp_space["degree"]["log"])
        
        try:
            if args.method == "esvm":
                perf_json = svm_function.cv_esvm_perf(data_x=x, 
                                                      data_y=self.y, 
                                                      classifier=classifier_name,
                                                      kernel=kernel,
                                                      C=C,
                                                      gamma=gamma,
                                                      coef0=coef0,
                                                      degree=degree,
                                                      nu=nu,
                                                      size=args.size,
                                                      max_iter=args.max_iter,
                                                      log=False,
                                                      fold=args.fold
                                                      )
            elif args.method == "svm":
                perf_json = svm_function.cv_svm_perf(data_x=x, 
                                                     data_y=self.y, 
                                                     classifier=classifier_name,
                                                     kernel=kernel,
                                                     C=C,
                                                     gamma=gamma,
                                                     coef0=coef0,
                                                     degree=degree,
                                                     nu=nu,
                                                     max_iter=args.max_iter,
                                                     log=False,
                                                     fold=args.fold
                                                     )
                
            score = perf_json["avg %s" % args.performance_value]
            self._perf = perf_json
        except Exception as e: 
            print(e, "error return -1.")
            score = -1
        print(score)
        return score


    def callback(self, study, trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            self.best_perf = self._perf
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                trial.number,
                trial.value,
                trial.params,
                )
            )
        # if study.best_trial == trial:
        #     self.best_perf = self._perf
            
        

objective = Objective(x=x, y=y, hp_space=hp_space, feature_selection=args.feature_selection)

storage = None
if not args.storage is None:
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(args.storage),
    )
    storage_study_name = [storage.get_study_name_from_id(temp_study._study_id) for temp_study in storage.get_all_studies()]
    
if not args.study_name is None and args.study_name in storage_study_name:
    study = optuna.load_study(study_name=args.study_name, storage=storage, sampler=study_sampler)
else:
    study = optuna.create_study(study_name=args.study_name, direction="maximize", storage=storage, sampler=study_sampler)
    
try:
    study.optimize(objective, n_trials=args.num_evals, timeout=args.timeout, n_jobs=args.n_jobs, gc_after_trial=True, show_progress_bar=True, callbacks=[objective.callback])
except optuna.exceptions.TrialPruned as e:
    print("Trial was pruned due to timeout.")

print("output=%s" % args.output)

with open('%s.json' % (args.output), 'w') as fp:
    json.dump(objective.best_perf, fp)

print("total time: %2.f" % (time.time() - total_time))
