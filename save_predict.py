import json
import argparse

import numpy as np
import pandas as pd

# from sklearnex import patch_sklearn 
# patch_sklearn()

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output', type=str, help='output file .json')
parser.add_argument('-m', '--method', default='svm', type=str, help='svm, libsvm, esvm')
parser.add_argument('-p', '--hp', default='hyper_parameter.csv', type=str, help='hyper_parameter file')
args = parser.parse_args()

print("input file: %s" % (args.input))
x = np.load(args.input)
print("label file: %s" % (args.label))
y = np.load(args.label)
print("hyper_parameter file: %s" % (args.hp))
if ".json" in args.hp:
    f = open(args.hp)
    hp = json.load(f)
    f.close()
else:
    df = pd.read_csv(args.hp)
    hp = df.iloc[0]

def cv_svm_perd_proba(data_x, data_y, fold=5, classifier='SVC', kernel='linear', C=0, gamma=0, degree=0, coef0=0, nu=0.5, max_iter=1e7, log=False):
    cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, fold)
    
    train_pred_list = []
    train_pred_proba_list = []
    train_y_list = []
    test_pred_list = []
    test_pred_proba_list = []
    test_y_list = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        model = svm_function.svm_train_model(x_train, y_train, 
                                             classifier=classifier, 
                                             kernel=kernel, 
                                             C=C, 
                                             gamma=gamma, 
                                             degree=degree, 
                                             coef0=coef0, 
                                             nu=nu, 
                                             max_iter=max_iter,
                                             log=log)
        
        train_pred_list.append(list(np.array(model.predict(x_train)).tolist()))
        train_pred_proba = np.array(model.predict_proba(x_train)).tolist()
        train_pred_proba_list.append(list(train_pred_proba))
        train_y_list.append(list(y_train.tolist()))
        
        test_pred_list.append(list(np.array(model.predict(x_test)).tolist()))
        test_pred_proba = np.array(model.predict_proba(x_test)).tolist()
        test_pred_proba_list.append(list(test_pred_proba))
        test_y_list.append(list(y_test.tolist()))
    
    
    return train_pred_list, train_pred_proba_list, train_y_list, test_pred_list, test_pred_proba_list, test_y_list
     

if args.method == 'svm':
    print("classifier=%s, kernel=%s, C=%s, gamma=%s, degree=%s, coef0=%s, nu=%s, max_iter=%s, log=%s" % (
        hp["classifier"], hp["kernel"], hp["C"], hp["gamma"], hp["degree"], hp["coef0"], hp["nu"], hp["max_iter"], hp["log"]))
    train_pred, train_proba, train_y, test_pred,test_proba, test_y = cv_svm_perd_proba(x, y,
                                                                 fold=hp["fold"],
                                                                 classifier=hp["classifier"],
                                                                 kernel=hp["kernel"],
                                                                 C=hp["C"],
                                                                 gamma=hp["gamma"],
                                                                 degree=hp["degree"],
                                                                 coef0=hp["coef0"],
                                                                 nu=hp["nu"],
                                                                 max_iter=hp["max_iter"],
                                                                 log=hp["log"],
                                                                 )

json_dict = {
    "train_pred": train_pred,
    "train_proba": train_proba,
    "train_y": train_y,
    "test_pred": test_pred,
    "test_proba": test_proba,
    "test_y": test_y,
    "input": args.input,
    "label": args.label,
    "hp": args.hp
}

with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dict, fp)
