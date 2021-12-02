import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn import metrics

from libsvm.svmutil import *
from imblearn.over_sampling import SMOTE

k_array = [1, 2, 3, 4]
power_array = [8, 10]
nor_array = [2, 3]
num_array = [1, 2, 3, 5, 8, 10]

data_y = np.load("data/linear_features/data_y.npy")

score_history = None
start_time = time.time()
for k in k_array:
    for power in power_array:
        for nor in nor_array:
            for num in num_array:
                print("k=%s, power=%s, nor=%s, num=%s" % (k, power, nor, num))
                
                data_x = np.load("data/linear_features/point/k" + str(k) + "p" + str(power) + "nor" + str(nor) + "n" + str(num) + ".npy")
                data_x = data_x.reshape(data_x.shape[0],-1)
                print(data_x.shape)
                
                x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20, shuffle=True, random_state=12)
                
                clf = svm.SVC(kernel='linear', C=1, class_weight='balanced').fit(x_train, y_train)
                score = clf.score(x_test, y_test)
                y_pred = clf.predict(x_test)
                macro = f1_score(y_test, y_pred, average='macro')
                print("sklearn")
                print("score:", score)
                print("macro:", macro)
                print(metrics.confusion_matrix(y_test, y_pred) / len(y_test))
                if macro >= 0.6:
                    with open("data/linear_features/model/k" + str(k) + "p" + str(power) + "nor" + str(nor) + "n" + str(num) + "point_1122_" + str(start_time) + ".pickle", 'wb') as f:
                        pickle.dump(clf, f)
                
                if score_history is None:
                    score_history = np.array([[k, power, nor, num, data_x.shape[1], score, macro]])
                else:
                    score_history = np.append(score_history, [[k, power, nor, num, data_x.shape[1], score, macro]], axis=0)
print(score_history)
print("All time:", time.time() - start_time)
print("data/linear_features/score_history/dle_point_score_history_" + str(start_time))
np.save("data/linear_features/score_history/dle_point_score_history_" + str(start_time), score_history)