import sys
import time

import optunity
import optunity.metrics
import numpy as np
import pandas as pd

from Function import svm_function
from Function.ensemble_svm import ensemble_svm


argv_dict = {"f": 10, "size": 1, "num_evals": 10, "pmap": 1, "save_path": "hpo_ensemble_svm_test.csv"}

for i in range(1, len(sys.argv), 2):
    argv_dict[sys.argv[i]]
    if sys.argv[i] in ["f", "size", "num_evals", "pmap"]:
        temp = int(sys.argv[i + 1])
    else:
        temp = sys.argv[i + 1]
    argv_dict[sys.argv[i]] = temp

print(argv_dict["save_path"])

total_time = time.time()

x_cytosol = pd.read_csv("data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC_cytosol.csv").to_numpy()
x_nucleus = pd.read_csv("data/merge_data/k1234_PCPseDNCGa_TNCGa_PseDNC_nucleus.csv").to_numpy()
consensusclass = np.genfromtxt("data/r_output/consensusClass_k10.csv")

pmap10 = optunity.parallel.create_pmap(argv_dict["pmap"])

space = {'kernel': {
                    '00': {'C': [-10, 10]},
                    '01': {'logGamma': [-10, 10], 'C': [-10, 10], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '02': {'logGamma': [-10, 10], 'C': [-10, 10]},
                    # '02': {'logGamma': [1.2, 1.7], 'C': [9.1, 9.2]},
                    '03': {'logGamma': [-10, 10], 'C': [-10, 10], 'coef0': [-10, 10]},
                    '10': {'n': [0, 1]},
                    '11': {'logGamma': [-10, 10], 'n': [0, 1], 'degree': [1, 10], 'coef0': [-10, 10]},
                    '12': {'logGamma': [-10, 10], 'n': [0, 1]},
                    '13': {'logGamma': [-10, 10], 'n': [0, 1], 'coef0': [-10, 10]}
                    }
        }

fold = argv_dict["f"]
size = argv_dict["size"]

def class_cluster_ensemble_svm(kernel='00', C=0, logGamma=0, degree=0, coef0=0, n=0.5):
    auroc_array = []
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
            output_data = svm_function.cluster_sampler(train_cytosol_x, train_cytosol_consensuclass, size=len(train_nucleus_x))
            
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
        
        esvm.train(ensemble_data_x, ensemble_data_y, parameter=parameter)
        e_data_x_shape = ensemble_data_x.shape
        auroc, pred_y_score = esvm.test(test_x, test_y)
        auroc_array.append(auroc)
        print("fold=%i, AUROC=%.2f" % (fold, auroc))
    return sum(auroc_array)/len(auroc_array)

optimal_svm_pars, info, _ = optunity.maximize_structured(class_cluster_ensemble_svm, space, num_evals=argv_dict["num_evals"], pmap=pmap10)
print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)
print("total time: %2.f" % (time.time() - total_time))

df = optunity.call_log2dataframe(info.call_log)
df = df.sort_values(by=['value'], ascending=False)
df.to_csv(argv_dict["save_path"])
