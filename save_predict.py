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
    
    train_pred_list = []
    train_pred_proba_list = []
    train_y_list = []
    test_pred_list = []
    test_pred_proba_list = []
    test_y_list = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = svm_function.cv_train_test(cv_x, cv_y, i)
        model = svm_function.svm_train_model(x_train, y_train, 
                                             kernel=kernel, 
                                             C=C, 
                                             logGamma=logGamma, 
                                             degree=degree, 
                                             coef0=coef0, 
                                             n=n, 
                                             max_iter=max_iter)
        
        train_pred_list.append(list(model.predict(x_train)))
        train_pred_proba = np.array(model.predict_proba(x_train)).tolist()
        train_pred_proba_list.append(list(train_pred_proba))
        train_y_list.append(list(y_train))
        
        test_pred_list.append(list(model.predict(x_test)))
        test_pred_proba = np.array(model.predict_proba(x_test)).tolist()
        test_pred_proba_list.append(list(test_pred_proba))
        test_y_list.append(list(y_test))
    
    
    return train_pred_list, train_pred_proba_list, train_y_list, test_pred_list, test_pred_proba_list, test_y_list
     

if args.method == 'svm':
    print("kernel=%s, C=%s, logGamma=%s, degree=%s, coef0=%s, n=%s, max_iter=%s" % (
        hp["kernel"], hp["C"], hp["logGamma"], hp["degree"], hp["coef0"], hp["n"], hp["max_iter"]))
    train_pred, train_proba, train_y, test_pred,test_proba, test_y = cv_svm_perd_proba(x, y,
                                                                 fold=10,
                                                                 kernel=hp["kernel"],
                                                                 C=hp["C"],
                                                                 logGamma=hp["logGamma"],
                                                                 degree=hp["degree"],
                                                                 coef0=hp["coef0"],
                                                                 n=hp["n"],
                                                                 max_iter=hp["max_iter"]
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
