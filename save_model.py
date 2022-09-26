import pickle 
import argparse

import numpy as np
import pandas as pd

from sklearn import metrics
from Function import svm_function
from Function.ensemble_svm import ensemble_svm


parser = argparse.ArgumentParser()
parser.add_argument('-x', '--train_x', type=str, help='train_x file .npy')
parser.add_argument('-y', '--train_y', type=str, help='train_y file .npy')
parser.add_argument('-i', '--test_x', type=str, help='test_x file .npy')
parser.add_argument('-l', '--test_y', type=str, help='test_y file .npy')
# parser.add_argument('-o', '--output', default='output.png', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-p', '--hp', default='hyper_parameter.csv', type=str, help='hyper_parameter file')
parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('--save', default='model.pickle', type=str, help='save model path')
args = parser.parse_args()

print("train_x file: %s" % (args.train_x))
x = np.load(args.train_x)
print("train_y file: %s" % (args.train_y))
y = np.load(args.train_y)
print("test_x file: %s" % (args.test_x))
test_x = np.load(args.test_x)
print("test_y file: %s" % (args.test_y))
test_y = np.load(args.test_y)
print("hyper_parameter file: %s" % (args.hp))
df = pd.read_csv(args.hp)
hp = df.iloc[0]

def svm_train_model_hp_object(x, y, hp_object, max_iter=1e7):
    clf = svm_function.svm_train_model(x_train=x, y_train=y, kernel=hp_object['kernel'], C=hp_object['C'], logGamma=hp_object['logGamma'], degree=hp_object['degree'], coef0=hp_object['coef0'], n=None, max_iter=max_iter)
    return clf

model = svm_train_model_hp_object(x, y, df.iloc[0])
decision_values = model.decision_function(test_x)
# decision_values = np.where(np.isfinite(decision_values), decision_values, 0) 
roc_score = metrics.roc_auc_score(test_y, decision_values)
print("AUROC: %.2f" % (roc_score))

print("save model file: %s" % (args.save))
with open(args.save, 'wb') as f:
    pickle.dump(model, f)