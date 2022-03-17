import sys
import time

import numpy as np
import pandas as pd
import optunity
import optunity.metrics

from os import listdir
from os.path import join
from os.path import isfile

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

space = {'kernel': {
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

argv_dict = {"f": 10, "size": 1, "num_evals": 10, "pmap": 1, "save_path": "hpo_ensemble_svm_test.csv"}

for i in range(1, len(sys.argv), 2):
    argv_dict[sys.argv[i]]
    if sys.argv[i] in ["f", "size", "num_evals", "pmap"]:
        temp = int(sys.argv[i + 1])
    else:
        temp = sys.argv[i + 1]
    argv_dict[sys.argv[i]] = temp

fold = argv_dict["f"]
size = argv_dict["size"]
num_evals = argv_dict["num_evals"]
pmap10 = optunity.parallel.create_pmap(argv_dict["pmap"])

total_time = time.time()

def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    x_train, y_train = svm_function.ensemble_data(x_train, y_train, size=size)
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

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='00', C=0, logGamma=0, degree=0, coef0=0, n=0.5):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n)
    roc_score, pred_score = model.test(x_test, y_test)
    return roc_score


data_x = pd.read_csv("data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC.csv").to_numpy()
data_y = np.load("data/data_y_train.npy")

cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=fold)
cv_svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, space, num_evals=num_evals, pmap=pmap10)
            
print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)
print("total time: %2.f" % (time.time() - total_time))

df = optunity.call_log2dataframe(info.call_log)
df = df.sort_values(by=['value'], ascending=False)
df.to_csv(argv_dict["save_path"])