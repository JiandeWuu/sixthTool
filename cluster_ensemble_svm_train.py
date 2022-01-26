import sys
import math
import time

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

total_time = time.time()

x_cytosol = pd.read_csv("data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC_cytosol.csv").to_numpy()
x_nucleus = pd.read_csv("data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC_nucleus.csv").to_numpy()
consensusclass = np.genfromtxt("data/r_output/consensusClass_k10.csv")

def cluster_sampler(data, cluster_class, size=1):
    class_nums, class_counts = np.unique(cluster_class, return_counts=True)
    l = len(data)
    output_data = None
    for i in range(len(class_nums)):
        class_idx = np.arange(l)[cluster_class == class_nums[i]]
        n = math.ceil(size / l * len(class_idx))
        np.random.shuffle(class_idx)
        if output_data is None:
            output_data = data[class_idx[:n], :]
        else:
            output_data = np.append(output_data, data[class_idx[:n], :], axis=0)
    return output_data


argv_dict = {"f": 10, "size": 100, "seed": None}

for i in range(1, len(sys.argv), 2):
    argv_dict[sys.argv[i]]
    temp = int(sys.argv[i + 1])
    argv_dict[sys.argv[i]] = temp

if not argv_dict['seed'] is None:
    np.random.seed(argv_dict['seed'])

auroc_array = []
pred_y_score_array = []

fold = argv_dict['f']
size = argv_dict['size']

cv_x_cytosol, cv_consensuclass = svm_function.CV_balanced(x_cytosol, consensusclass, fold)
cv_x_nucleus, cv_y_nucleus = svm_function.CV_balanced(x_nucleus, np.zeros((len(x_nucleus))), fold)

for i in range(fold):
    print("fold:", i)
    train_cytosol_x, train_cytosol_consensuclass, test_cytosol_x, _ = svm_function.cv_train_test(cv_x_cytosol, cv_consensuclass, i)
    train_nucleus_x, train_nucleus_y, test_nucleus_x, test_nucleus_y = svm_function.cv_train_test(cv_x_nucleus, cv_y_nucleus, i)
    
    test_x = np.append(test_cytosol_x, test_nucleus_x, axis=0)
    test_y = np.append(np.ones(len(test_cytosol_x)), np.zeros(len(test_nucleus_x)), axis=0)
    
    ensemble_data_x = None
    ensemble_data_y = None
    for j in range(size):
        output_data = cluster_sampler(train_cytosol_x, train_cytosol_consensuclass, size=len(train_nucleus_x))
        
        output_y = np.array([np.append(np.ones(len(output_data)), np.zeros(len(train_nucleus_x)))])
        
        output_data = np.append(output_data, train_nucleus_x, axis=0)
        output_data = np.array([output_data])
        
        if ensemble_data_x is None:
            ensemble_data_x = output_data
            ensemble_data_y = output_y
        else:
            ensemble_data_x = np.append(ensemble_data_x, output_data, axis=0)
            ensemble_data_y = np.append(ensemble_data_y, output_y, axis=0)
    
    esvm = ensemble_svm()
    esvm.train(ensemble_data_x, ensemble_data_y, parameter="-s 1 -t 0")
    e_data_x_shape = ensemble_data_x.shape
    auroc, pred_y_score = esvm.test(test_x, test_y)
    # auroc = metrics.roc_auc_score(test_y, pred_y)
    auroc_array.append(auroc)
    pred_y_score_array.append(pred_y_score)


print("pred_y_score_array:", pred_y_score_array)
print("auroc:", auroc_array)
print("size:", size)
print("auroc avg:", sum(auroc_array) / len(auroc_array))
print("total time: %.2f s" % (time.time() - total_time))
