import time

import numpy as np

from os import listdir
from os.path import join
from os.path import isfile

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

size = 500

# dir_path = "data/Pse_in_One2/DNA/train/"
dir_path = "data/k_mers/train/"
# dir_path = "data/linear_features/linear/"
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

data_y = np.load("data/data_y_train.npy")

log_array = []
start_time = time.time()
for file_name in onlyfiles:
    if file_name.split(".")[-1] == "npy":
        file_time = time.time()
        
        # data_x = np.genfromtxt(dir_path + file_name, delimiter=',')
        data_x = np.load(dir_path + file_name)
        # data_x = data_x.reshape(data_x.shape[0],-1)

        if data_x.shape[1] < 400:
            cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, 10)
        
            roc_score_array = []
            for i in range(10):
                train_x, train_y, test_x, test_y = svm_function.cv_train_test(cv_x, cv_y, i)
                train_x, train_y = svm_function.ensemble_data(train_x, train_y, size=size)
                
                esvm = ensemble_svm()
                esvm.train(train_x, train_y, parameter="-s 1 -t 0")
                roc_score, pred_score = esvm.test(test_x, test_y)
                roc_score_array.append(roc_score)

            auroc = sum(roc_score_array) / len(roc_score_array)
            log_array.append([file_name, train_x.shape, time.time() - file_time, auroc])
            print("file name=%s, time=%.2f" % (file_name, time.time() - file_time))
print(log_array)
np.save("data/log/kmers_train_cv_msvm_s1t0_f10s500.npy", np.array(log_array))
