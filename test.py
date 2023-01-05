import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

from Function import svm_function

X = np.load("data/merge_data/0916chiu/classifier/nuc_lncRNA__cyto_lncRNA_train.npy")
y = np.load("data/merge_data/0916chiu/classifier/nuc_lncRNA__cyto_lncRNA_train_y.npy")

x_train, y_train = svm_function.ensemble_data(X, y, 10)

feature_rank = None
for i in range(len(x_train)):
    # define RFE
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=1)
    # fit RFE
    rfe.fit(x_train[i], y_train[i])
    if feature_rank is None:
        feature_rank = np.array(rfe.ranking_)
    else:
        feature_rank += np.array(rfe.ranking_)

order = feature_rank.argsort()
ranks = order.argsort()

f = open("/data/jand/sixthTool/data/merge_data/0916chiu/classifier/output/nuc_lncRNA__cyto_lncRNA_esvmf10s10p4e512_v2.json")
hp = json.load(f)
f.close()

perf_array = []
# for i in range(10):
for i in range(len(ranks)):
    print(i, X[:, ranks <= i].shape)
    json_dcit = svm_function.cv_esvm_perf(X[:, ranks <= i], y, 
                                      fold=10, 
                                      kernel=hp["kernel"], 
                                      C=hp["C"], 
                                      logGamma=hp["logGamma"], 
                                      degree=hp["degree"], 
                                      coef0=hp["coef0"], 
                                      n=hp["n"], 
                                      size=hp["size"], 
                                      max_iter=hp["max_iter"])
    perf_array.append([i + 1, 
                       json_dcit['avg Accy'],
                       json_dcit['avg Recall'],
                       json_dcit['avg Prec'],
                       json_dcit['avg Spec'],
                       json_dcit['avg Npv'],
                       json_dcit['avg F1sc'],
                       json_dcit['avg AUROC']
                       ])

pd.DataFrame(perf_array, columns=["rank cutoff",
                                  "avg Accy",
                                  "avg Recall",
                                  "avg Prec",
                                  "avg Spec",
                                  "avg Npv",
                                  "avg F1sc",
                                  "avg Accy"]).to_csv("/data/jand/sixthTool/data/merge_data/0916chiu/classifier/output/nuc_lncRNA__cyto_lncRNA_esvmf10s10p4e512_v2__Feature_rank.csv")