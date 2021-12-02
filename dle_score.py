import time

import numpy as np
import pandas as pd

from Features.dle import dle

from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score

# k_array = [1]
# power_array = [4]
# nor_array = [0]

k_array = [1, 2, 3]
power_array = [10]
nor_array = [2, 3]
output_array = [4, 5, 10]

df = pd.read_csv("data/linear_features/cdhit80_data_seq_loc75_train.csv")
data_y = np.where(df["loc"].to_numpy() == "Cytosolic", 1, 0)

score_history = None

for k in k_array:
    for power in power_array:
        for nor in nor_array:
            print("k=%s, power=%s, nor=%s" % (k, power, nor))
            data_x, vocab = dle(df["Sequence"], k=k, power=power, normalized=nor)
            
            np.save("data/linear_features/cdhit80_loc75_k" + str(k) + "_power" + str(power) + "_nor" + str(nor) + "_output" + str(output), np.append(np.array([data_y]).T, data_x, axis=1))
            
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20, shuffle=True, random_state=12)

            print("x_train", x_train.shape)
            print("x_test", x_test.shape)
            print("y_test", np.unique(y_test, return_counts=True))

            clf = svm.SVC(kernel='rbf', C=1, class_weight='balanced').fit(x_train, y_train)
            
            score = clf.score(x_test, y_test)
            print("score:", score)
            if score_history is None:
                score_history = np.array([[k, power, nor, score]])
            else:
                score_history = np.append(score_history, [[k, power, nor, score]], axis=0)
print(score_history)
np.save("data/linear_features/score_history/score_history", score_history)
