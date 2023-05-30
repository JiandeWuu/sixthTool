import json
import argparse

import numpy as np
import pandas as pd

from Function import svm_function
from Function.ensemble_svm import ensemble_svm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input file .npy')
parser.add_argument('-l', '--label', type=str, help='label file .npy')
parser.add_argument('-o', '--output', default='output', type=str, help='output file .npy')
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

def cv_svm_perd_proba(data_x, data_y, fold=5, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=1e7):
    cv_x, cv_y = svm_function.CV_balanced(data_x, data_y, fold)
    
    train_pred_proba_list = []
    train_y_list = []
    test_pred_proba_list = []
    test_y_list = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        model = svm_function.svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n, max_iter=max_iter)
        
        train_pred_proba = model.predict_proba(x_train)[:, 1]
        train_pred_proba_list.append(list(train_pred_proba))
        train_y_list.append(list(y_train))
        
        test_pred_proba = model.predict_proba(x_test)[:, 1]
        test_pred_proba_list.append(list(test_pred_proba))
        test_y_list.append(list(y_test))
    
    
    return train_pred_proba_list, train_y_list, test_pred_proba_list, test_y_list
     

if args.method == 'svm':
    train_proba, train_y, test_proba, test_y = cv_svm_perd_proba(x, y,
                                                                 fold=10,
                                                                 C=hp["C"],
                                                                 logGamma=hp["logGamma"],
                                                                 degree=hp["degree"],
                                                                 coef0=hp["coef0"],
                                                                 n=hp["n"],
                                                                 max_iter=hp["max_iter"]
                                                                 )


json_dict = {
    "train_proba": train_proba,
    "train_y": train_y,
    "test_proba": test_proba,
    "test_y": test_y,
    "input": args.input,
    "label": args.label,
    "hp": hp
}

with open('%s.json' % (args.output), 'w') as fp:
    json.dump(json_dict, fp)
