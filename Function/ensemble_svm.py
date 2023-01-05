import time
import math

import numpy as np

from sklearn import metrics
# from libsvm.svmutil import svm_problem
# from libsvm.svmutil import svm_parameter
# from libsvm.svmutil import svm_train
# from libsvm.svmutil import svm_predict


from . import svm_function

class ensemble_svm():
    def __init__(self):
        self.model_array = None
        self.model_size = None
    
    def train(self, data, label, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=1e7):
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
            
            m = svm_function.svm_train_model(d, l, kernel, C, logGamma, degree, coef0, n, max_iter=max_iter)
            
            model_array.append(m)
            decision_values = m.decision_function(d)
            
            try:
                print("auroc:", metrics.roc_auc_score(l, decision_values))
            except:
                print("error return 0.5.")
            print("ensemble svm step: %s/%s | %.2fs" % (i, self.model_size, time.time() - step_time))
        
        self.model_array = model_array
        print("ensemble svm train: %.2fs" % (time.time() - train_time))
        
        return None

    def test(self, x, y):
        output = None
        for m in self.model_array:
            p_label = m.predict(x)
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
            
        return metrics.roc_auc_score(y, pred_y)
    
    def predict(self, x):
        output = None
        for m in self.model_array:
            p_label = m.predict(x)
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
