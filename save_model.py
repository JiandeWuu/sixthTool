from audioop import avg
import json
import pickle
import argparse

import _pickle as cPickle

import numpy as np
import pandas as pd


from sklearn import metrics
from Function import svm_function
from Function.ensemble_svm import ensemble_svm


parser = argparse.ArgumentParser()
parser.add_argument('-x', '--train_x', type=str, help='train_x file .npy')
parser.add_argument('-y', '--train_y', type=str, help='train_y file .npy')
parser.add_argument('-i', '--test_x', default=None, type=str, help='test_x file .npy')
parser.add_argument('-l', '--test_y', default=None, type=str, help='test_y file .npy')
# parser.add_argument('-o', '--output', default='output.png', type=str, help='output file')
parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-p', '--hp', default='hyper_parameter.csv', type=str, help='hyper_parameter file')
# parser.add_argument('-s', '--size', default=1, type=int, help='Ensemble SVM size')
parser.add_argument('-s', '--save', default='model.pickle', type=str, help='save model path')
args = parser.parse_args()

print("train_x file: %s" % (args.train_x))
x = np.load(args.train_x)
print("train_y file: %s" % (args.train_y))
y = np.load(args.train_y)
if args.test_x and args.test_y:
    print("test_x file: %s" % (args.test_x))
    test_x = np.load(args.test_x)
    print("test_y file: %s" % (args.test_y))
    test_y = np.load(args.test_y)
print("hyper_parameter file: %s" % (args.hp))
if ".json" in args.hp:
    f = open(args.hp)
    hp = json.load(f)
    f.close()
else:
    df = pd.read_csv(args.hp)
    hp = df.iloc[0]

def esvm_train_model_hp_object(x, y, hp_object):
    clf = svm_function.esvm_train_model(x_train=x, y_train=y, 
                                        classifier=hp_object['classifier'], 
                                        kernel=hp_object['kernel'], 
                                        C=hp_object['C'], 
                                        gamma=hp_object['gamma'], 
                                        degree=hp_object['degree'], 
                                        coef0=hp_object['coef0'], 
                                        nu=hp_object['nu'], 
                                        size=hp_object['size'], 
                                        max_iter=hp_object['max_iter'],
                                        log=hp_object['log'])
    return clf

def svm_train_model_hp_object(x, y, hp_object):
    clf = svm_function.svm_train_model(x_train=x, y_train=y, 
                                        kernel=hp_object['kernel'], 
                                        C=hp_object['C'], 
                                        logGamma=hp_object['logGamma'], 
                                        degree=hp_object['degree'], 
                                        coef0=hp_object['coef0'], 
                                        n=hp_object['n'], 
                                        max_iter=hp_object['max_iter'])
    return clf

if args.method == 'svm':
    train_def = svm_train_model_hp_object
    model = train_def(x, y, hp)
    if args.test_x and args.test_y:
        decision_values = model.decision_function(test_x)
        # decision_values = np.where(np.isfinite(decision_values), decision_values, 0) 
        roc_score = metrics.roc_auc_score(test_y, decision_values)
        print("AUROC: %.2f" % (roc_score))
elif args.method == 'esvm':
    train_def = esvm_train_model_hp_object
    model = train_def(x, y, hp)
    if args.test_x and args.test_y:
        roc_score = model.test(test_x, test_y)
        print("AUROC: %.2f" % (roc_score))

print("save model file: %s" % (args.save))
with open(args.save, 'wb') as f:
    if args.method == 'svm':
        pickle.dump(model, f)
    elif args.method == 'esvm':
        cPickle.dump(model, f)
