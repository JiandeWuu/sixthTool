import time
import math

import numpy as np
import pandas as pd

from os import listdir
from os.path import join
from os.path import isfile


from sklearn import metrics

from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict

class multi_SVM():
    def __init__(self, class_weight='None'):
        self.class_weight = class_weight
        self.model_array = None
    
    def train(self, x, y, size=1, parameter=""):
        if self.class_weight == 'balanced':
            data = None
            label = None
            for i in range(size):
                d, l = self.balanced_data(x, y)
                if data is None:
                    data = d
                    label = l
                else:
                    data = np.append(data, d, axis=0)
                    label = np.append(label, l, axis=0)
        
        model_array = []
        for d, l in zip(data, label):
            arr = np.arange(len(l))
            np.random.shuffle(arr)
            d = d[arr]
            l = l[arr]

            prob = svm_problem(l, d)
            param = svm_parameter(parameter)
            m = svm_train(prob, param)
            model_array.append(m)
        
        self.model_array = model_array
        
        return None
    
    def test(self, x, y):
        output = None
        for m in self.model_array:
            p_label, p_acc, p_val = svm_predict(y, x, m)
            if output is None:
                output = np.array([p_label])
            else:
                output = np.append(output, np.array([p_label]), axis=0)
        
        pred_y = []
        for o in output.T:
            u, c = np.unique(o, return_counts=True)
            pred_y.append(u[c == c.max()][0])
        
        return metrics.roc_auc_score(y, pred_y)
    
    def predict(self, x):
        output = None
        for m in self.model_array:
            p_label, p_acc, p_val = svm_predict([], x, m)
            if output is None:
                output = np.array([p_label])
            else:
                output = np.append(output, np.array([p_label]), axis=0)
        
        pred_y = []
        for o in output.T:
            u, c = np.unique(o, return_counts=True)
            pred_y.append(u[c == c.max()][0])
        
        return pred_y
    
    def balanced_data(self, x, y):
        unique, count = np.unique(y, return_counts=True)
        min_count = min(count)
        big_batch = math.ceil(max(count) / min_count)
        min_u = np.where(count == min_count, unique, -1)
        
        data = None
        label = None
        for u in unique:
            u_data = None
            if not u in min_u:
                while u_data is None or len(u_data) != big_batch * min_count:
                    x_u = x[y == u]
                    np.random.shuffle(x_u)
                    mod = len(x_u) % min_count
                    if mod != 0:
                        arr = np.arange(len(x_u) - mod)
                        np.random.shuffle(arr)
                        x_u = np.append(x_u, x_u[arr[mod - min_count:]], axis=0)
                    if u_data is None:
                        u_data = x_u
                    else:
                        u_data = np.append(u_data, x_u[:(big_batch * min_count) - len(u_data)], axis=0)
                u_data = np.array(np.split(u_data, big_batch))
            else:
                x_u = x[y == u]
                x_u = np.expand_dims(x_u, axis=0)
                u_data = np.repeat(x_u, big_batch, axis=0)

            if data is None:
                data = u_data
                label = np.full((big_batch, min_count), u)
            else:
                data = np.append(data, u_data, axis=1)  
                label = np.append(label, np.full((big_batch, min_count), u), axis=1)

        return data, label

def CV(x, y, folder):
    unique, count = np.unique(y, return_counts=True)
    cv_x = []
    cv_y = []
    for u in unique:
        u_x = x[y == u]
        u_y = y[y == u]
        arr = np.arange(len(u_x))
        np.random.shuffle(arr)
        u_x = u_x[arr]
        u_y = u_y[arr]
        
        linspace = np.linspace(0, len(u_x), folder + 1, dtype=int)
        
        for i in range(folder):
            if unique[0] == u:
                cv_x.append(u_x[linspace[i]:linspace[i+1]])
                cv_y.append(u_y[linspace[i]:linspace[i+1]])
            else:
                cv_x[i] = np.append(cv_x[i], u_x[linspace[i]:linspace[i+1]], axis=0)
                cv_y[i] = np.append(cv_y[i], u_y[linspace[i]:linspace[i+1]], axis=0)
    return cv_x, cv_y

def cv_msvm_score(x, y, folder, size=15, parameter=""):
    cv_x, cv_y = CV(x, y, folder)
    score_array = []
    for j in range(len(cv_x)):
        train_x = None
        for i in range(len(cv_x)):
            if i == j :
                test_x = cv_x[i]
                test_y = cv_y[i]
            else:
                if train_x is None:
                    train_x = cv_x[i]
                    train_y = cv_y[i]
                else:
                    train_x = np.append(train_x, cv_x[i], axis=0)
                    train_y = np.append(train_y, cv_y[i], axis=0)
        msvm = multi_SVM(class_weight='balanced')
        msvm.train(train_x, train_y, size=size, parameter=parameter)
        score_array.append(msvm.test(test_x, test_y))
    return score_array


dir_path = "data/Pse_in_One2/DNA/train/"
# dir_path = "data/k_mers/"
# dir_path = "data/linear_features/linear/"
onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

data_y = np.load("data/data_y_train.npy")

log_array = []
start_time = time.time()
for file_name in onlyfiles:
    if file_name.split(".")[-1] == "csv":
        file_time = time.time()
        
        data_x = np.genfromtxt(dir_path + file_name, delimiter=',')
        # data_x = np.load(dir_path + file_name)
        # data_x = data_x.reshape(data_x.shape[0],-1)

        if data_x.shape[1] < 400:
            score_array = cv_msvm_score(data_x, data_y, 10, size=100, parameter="")
            cv_msvm_auroc = sum(score_array) / len(score_array)
            
            log_array.append([file_name, data_x.shape[1], time.time() - file_time, cv_msvm_auroc])
            print("file name=%s, time=%.2f" % (file_name, time.time() - file_time))

print(log_array)
np.save("data/log/pseinone_train_cv_msvm_f10s100.npy", np.array(log_array))
