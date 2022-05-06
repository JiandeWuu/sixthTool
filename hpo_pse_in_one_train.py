import time

import numpy as np
import optunity
import optunity.metrics

from os import listdir
from os.path import join
from os.path import isfile

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

space = {'kernel': {'00': {'C': [-10, 10]},
                    '01': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '02': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    '03': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    '10': {'n': [0, 1]},
                    '11': {'logGamma': [-10, 10], 'n': [0, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '12': {'logGamma': [-10, 10], 'n': [0, 1]},
                    '13': {'logGamma': [-10, 10], 'n': [0, 1], 'coef0': [-10, 10]}
                    }
        }

size = 100
num_evals = 100

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


dir_path = "data/Pse_in_One2/DNA_0320/train/"
# dir_path = "data/k_mers/train/"
# dir_path = "data/linear_features/linear/train/"
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

data_y = np.load("data/society/train_y_0320_loc75_01.npy")

log_array = []
start_time = time.time()
for file_name in onlyfiles:
    if file_name.split(".")[-1] == "npy":
        file_time = time.time()
        
        # data_x = np.genfromtxt(dir_path + file_name, delimiter=',')
        data_x = np.load(dir_path + file_name)
        # data_x = data_x.reshape(data_x.shape[0],-1)

        if data_x.shape[1] < 500:
            cv_decorator = optunity.cross_validated(x=data_x, y=data_y, num_folds=5)
            cv_svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

            optimal_svm_pars, info, _ = optunity.maximize_structured(cv_svm_tuned_auroc, space, num_evals=num_evals)
            print("File Name:" + file_name)
            print("Optimal parameters" + str(optimal_svm_pars))
            print("AUROC of tuned SVM: %1.3f" % info.optimum)
            log_array.append([file_name, time.time() - file_time, optimal_svm_pars, info.optimum])
            
print(log_array)
np.save("data/hpo/pse_in_one_0320_train.npy", np.array(log_array))

print("ALL TIME: %d" % (time.time() - start_time))
