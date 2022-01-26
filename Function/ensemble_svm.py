import time
import math

import numpy as np

from sklearn import metrics
from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict

class ensemble_svm():
    def __init__(self):
        self.model_array = None
        self.model_size = None
    
    def train(self, data, label, parameter=""):
        train_time = time.time()
        model_array = []
        self.model_size = len(data)
        
        i = 0
        for d, l in zip(data, label):
            step_time = time.time()
            i += 1
            arr = np.arange(len(l))
            np.random.shuffle(arr)
            d = d[arr]
            l = l[arr]

            prob = svm_problem(l, d)
            param = svm_parameter(parameter)
            m = svm_train(prob, param)
            model_array.append(m)
            p_label, p_acc, p_val = svm_predict(l, d, m)
            
            print("auroc:", metrics.roc_auc_score(l, p_label))
            print("ensemble svm step: %s/%s | %.2fs" % (i, self.model_size, time.time() - step_time))
        
        self.model_array = model_array
        print("ensemble svm train: %.2fs" % (time.time() - train_time))
        
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
        pred_y_score = []
        for o in output.T:
            u, c = np.unique(o, return_counts=True)
            pred_y.append(u[c == c.max()][0])
            pred_y_score.append(c.max() / sum(c))
            
        return metrics.roc_auc_score(y, pred_y), sum(pred_y_score) / len(pred_y_score)
    
    def predict(self, x):
        output = None
        for m in self.model_array:
            p_label, p_acc, p_val = svm_predict([], x, m)
            if output is None:
                output = np.array([p_label])
            else:
                output = np.append(output, np.array([p_label]), axis=0)

        pred_y = []
        pred_y_score = []
        for o in output.T:
            u, c = np.unique(o, return_counts=True)
            pred_y.append(u[c == c.max()][0])
            pred_y_score.append(c.max() / sum(c))
        
        return pred_y, pred_y_score
