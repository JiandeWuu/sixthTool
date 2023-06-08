import math

import numpy as np

from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from libsvm.svmutil import svm_problem
from libsvm.svmutil import svm_parameter
from libsvm.svmutil import svm_train
from libsvm.svmutil import svm_predict


from Function.ensemble_svm import ensemble_svm

def CV(x, y, fold, seed=None):
    if seed:
        np.random.seed(seed)
    
    arr = np.arange(len(y))
    np.random.shuffle(arr)
    x = x[arr]
    y = y[arr]
    
    cv_x = []
    cv_y = []
    linspace = np.linspace(0, len(y), fold + 1, dtype=int)
    for i in range(fold):
        cv_x.append(x[linspace[i]:linspace[i+1]])
        cv_y.append(y[linspace[i]:linspace[i+1]])
    return cv_x, cv_y

def CV_balanced(x, y, fold):
    unique, count = np.unique(y, return_counts=True)
    cv_x = []
    cv_y = []
    for u in unique:
        u_x = x[y == u]
        u_y = y[y == u]
        arr = np.arange(len(u_x))
        np.random.shuffle(arr)
        u_x = u_x[arr]
        u_y = u_y[arr]
        linspace = np.linspace(0, len(u_x), fold + 1, dtype=int)
        
        for i in range(fold):
            if unique[0] == u:
                cv_x.append(u_x[linspace[i]:linspace[i+1]])
                cv_y.append(u_y[linspace[i]:linspace[i+1]])
            else:
                cv_x[i] = np.append(cv_x[i], u_x[linspace[i]:linspace[i+1]], axis=0)
                cv_y[i] = np.append(cv_y[i], u_y[linspace[i]:linspace[i+1]], axis=0)
    return cv_x, cv_y

def cv_train_test(cv_x, cv_y, test_fold=1):
    train_x = None
    for i in range(len(cv_x)):
        if i == test_fold :
            test_x = cv_x[i]
            test_y = cv_y[i]
        else:
            if train_x is None:
                train_x = cv_x[i]
                train_y = cv_y[i]
            else:
                train_x = np.append(train_x, cv_x[i], axis=0)
                train_y = np.append(train_y, cv_y[i], axis=0)
    return train_x, train_y, test_x, test_y

# def cv_msvm_score(x, y, folder, size=15, parameter=""):
#     cv_x, cv_y = CV(x, y, folder)
#     score_array = []
#     for j in range(len(cv_x)):
#         train_x = None
#         for i in range(len(cv_x)):
#             if i == j :
#                 test_x = cv_x[i]
#                 test_y = cv_y[i]
#             else:
#                 if train_x is None:
#                     train_x = cv_x[i]
#                     train_y = cv_y[i]
#                 else:
#                     train_x = np.append(train_x, cv_x[i], axis=0)
#                     train_y = np.append(train_y, cv_y[i], axis=0)
#         msvm = ensemble_SVM(class_weight='balanced')
#         msvm.train(train_x, train_y, size=size, parameter=parameter)
#         score_array.append(msvm.test(test_x, test_y))
#     return score_array

def balanced_data(x, y):
    unique, count = np.unique(y, return_counts=True)
    min_count = min(count)
    big_batch = math.ceil(max(count) / min_count)
    min_u = np.where(count == min_count, unique, -1)
    
    data = None
    label = None
    for u in unique:
        u_data = None
        if not u in min_u:
            while u_data is None or len(u_data) != big_batch * min_count:
                x_u = x[y == u]
                np.random.shuffle(x_u)
                mod = len(x_u) % min_count
                if mod != 0:
                    arr = np.arange(len(x_u) - mod)
                    np.random.shuffle(arr)
                    x_u = np.append(x_u, x_u[arr[mod - min_count:]], axis=0)
                if u_data is None:
                    u_data = x_u
                else:
                    u_data = np.append(u_data, x_u[:(big_batch * min_count) - len(u_data)], axis=0)
            # u_data = np.array(np.split(u_data, big_batch))
            u_data = u_data.reshape(big_batch, min_count, -1)
        else:
            x_u = x[y == u]
            x_u = np.expand_dims(x_u, axis=0)
            u_data = np.repeat(x_u, big_batch, axis=0)

        if data is None:
            data = u_data
            label = np.full((big_batch, min_count), u)
        else:
            data = np.append(data, u_data, axis=1)  
            label = np.append(label, np.full((big_batch, min_count), u), axis=1)

    return data, label

def ensemble_data(x, y, size=1):
    data = None
    label = None
    for i in range(size):
        d, l = balanced_data(x, y)
        if data is None:
            data = d
            label = l
        else:
            data = np.append(data, d, axis=0)
            label = np.append(label, l, axis=0)
    return data, label

def cluster_sampler(data, cluster_class, size=1):
    class_nums, class_counts = np.unique(cluster_class, return_counts=True)
    l = len(data)
    output_data = None
    for i in range(len(class_nums)):
        class_idx = np.arange(l)[cluster_class == class_nums[i]]
        n = math.ceil(size / l * len(class_idx))
        np.random.shuffle(class_idx)
        if output_data is None:
            output_data = data[class_idx[:n], :]
        else:
            output_data = np.append(output_data, data[class_idx[:n], :], axis=0)
    return output_data

def esvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n, size, max_iter):
    """A generic eeSVM training function, with arguments based on the chosen kernel."""
    x_train, y_train = ensemble_data(x_train, y_train, size=size)
    esvm = ensemble_svm()
    esvm.train(x_train, y_train, kernel, C, logGamma, degree, coef0, n, max_iter=max_iter)
    
    return esvm

def cv_esvm_perf(data_x, data_y, fold=5, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, size=1, max_iter=1e7):
    cv_x, cv_y = CV_balanced(data_x, data_y, fold)
    
    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    npv_array = []
    f1sc_array = []
    auroc_array = []
    cm_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = cv_train_test(cv_x, cv_y, i)
        model = esvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n, size=size, max_iter=max_iter)
        
        # roc_score = model.test(x_test, y_test)
        y_train_pred, _ = model.predict(x_train)
        y_test_pred, y_test_score = model.predict(x_test)
        roc_score = metrics.roc_auc_score(y_test, y_test_score)
        auroc_array.append(roc_score)
        
        cm_array.append(np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]).tolist())
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        acc_array.append((tn + tp) / (tn + fp + fn + tp))
        recall_array.append(tp / (fn + tp))
        prec_array.append(tp / (fp + tp))
        spec_array.append(tn / (tn + fp))
        npv_array.append(tn / (tn + fn))
        f1sc_array.append(2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp))))
    
    json_dict = {
        "kernel": kernel, "C": C, "logGamma": logGamma, "degree": degree, "coef0": coef0, "n": n, "size":size, "max_iter":max_iter,
        "fold Accy": acc_array,
        "avg Accy": sum(acc_array) / len(acc_array),
        "std Accy": np.std(acc_array),
        "fold Recall": recall_array,
        "avg Recall": sum(recall_array) / len(recall_array),
        "std Recall": np.std(recall_array),
        "fold Prec": prec_array,
        "avg Prec": sum(prec_array) / len(prec_array),
        "std Prec": np.std(prec_array),
        "fold Spec": spec_array,
        "avg Spec": sum(spec_array) / len(spec_array),
        "std Spec": np.std(spec_array),
        "fold Npv": npv_array,
        "avg Npv": sum(npv_array) / len(npv_array),
        "std Npv": np.std(npv_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "std F1sc": np.std(f1sc_array),
        "fold AUROC": auroc_array,
        "avg AUROC": sum(auroc_array) / len(auroc_array),
        "std AUROC": np.std(auroc_array),
        "confusion matrix": cm_array
    }
    
    return json_dict
        

def libsvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, w0, w1, n):
    """A generic SVM training function, with arguments based on the chosen kernel."""

    parameter = "-s " + kernel[0] + " -t " + kernel[1] + " -w0 " + str(w0) + " -w1 " + str(w1)
    if kernel[0] == '0':
        parameter += " -c " + str(2 ** C)
    else:
        parameter += " -n " + str(n)
        
    if kernel[1] != '0':
        parameter += " -g " + str(2 ** logGamma)
    if kernel[1] == '1':
        parameter += " -d " + str(int(degree))
    if kernel[1] == '1' or kernel[1] == '3':
        parameter += " -r " + str(2 ** coef0)
    prob = svm_problem(y_train, x_train)
    param = svm_parameter(parameter)
    m = svm_train(prob, param)
    
    return m

def cv_libsvm_perf(data_x, data_y, fold=5, kernel='02', C=1, logGamma=1, degree=3, coef0=0, w0=1, w1=1, n=0.5):
    cv_x, cv_y = CV_balanced(data_x, data_y, fold)
    
    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    f1sc_array = []
    auroc_array = []
    cm_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = cv_train_test(cv_x, cv_y, i)
        model = libsvm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, w0, w1, n)
        
        y_train_pred, p_acc, p_val = svm_predict(y_train, x_train, model)
        y_test_pred, p_acc, p_val = svm_predict(y_test, x_test, model)
        
        p_val = np.where(np.isfinite(p_val), p_val, 0) 
        roc_score = metrics.roc_auc_score(y_test, p_val)
        auroc_array.append(roc_score)
        
        cm_array.append(np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]).tolist())
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        acc_array.append((tn + tp) / (tn + fp + fn + tp))
        recall_array.append(tp / (fn + tp))
        prec_array.append(tp / (fp + tp))
        spec_array.append(tn / (tn + fp))
        f1sc_array.append(2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp))))
    
    json_dict = {
        "fold Accy": acc_array,
        "avg Accy": sum(acc_array) / len(acc_array),
        "std Accy": np.std(acc_array),
        "fold Recall": recall_array,
        "avg Recall": sum(recall_array) / len(recall_array),
        "std Recall": np.std(recall_array),
        "fold Prec": prec_array,
        "avg Prec": sum(prec_array) / len(prec_array),
        "std Prec": np.std(prec_array),
        "fold Spec": spec_array,
        "avg Spec": sum(spec_array) / len(spec_array),
        "std Spec": np.std(spec_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "std F1sc": np.std(f1sc_array),
        "fold AUROC": auroc_array,
        "avg AUROC": sum(auroc_array) / len(auroc_array),
        "std AUROC": np.std(auroc_array),
        "confusion matrix": cm_array
    }
    
    return json_dict
        
   
def svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n, max_iter=1e7):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    kernel = kernel.split("_")
    if C:
        C = float(C)
    if logGamma:
        logGamma = float(logGamma)
    if degree:
        degree = int(degree)
    if coef0:
        coef0 = float(coef0)
    if n:
        n = float(n)
    
    if kernel[0] == "C":
        if kernel[1] == "linear":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "poly":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), degree=degree, coef0=(2 ** coef0), class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "rbf":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "sigmoid":
            clf = svm.SVC(kernel=kernel[1], C=(2 ** C), gamma=(2 ** logGamma), coef0=(2 ** coef0), class_weight='balanced', max_iter=max_iter, probability=True)
    elif kernel[0] == "Nu":
        if kernel[1] == "linear":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "poly":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), degree=degree, coef0=(2 ** coef0), class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "rbf":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), class_weight='balanced', max_iter=max_iter, probability=True)
        elif kernel[1] == "sigmoid":
            clf = svm.NuSVC(kernel=kernel[1], nu=n, gamma=(2 ** logGamma), coef0=(2 ** coef0), class_weight='balanced', max_iter=max_iter, probability=True)
    
    clf.fit(x_train, y_train)
    return clf

def cv_svm_perf(data_x, data_y, fold=5, kernel='C_linear', C=0, logGamma=0, degree=0, coef0=0, n=0.5, max_iter=1e7):
    cv_x, cv_y = CV_balanced(data_x, data_y, fold)
    
    acc_array = []
    recall_array = []
    prec_array = []
    spec_array = []
    npv_array = []
    f1sc_array = []
    auroc_array = []
    cm_array = []
    for i in range(fold):
        x_train, y_train, x_test, y_test = cv_train_test(cv_x, cv_y, i)
        model = svm_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0, n, max_iter=max_iter)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        y_test_pred_proba = model.predict_proba(x_test)
        
        # decision_values = model.decision_function(x_test)
        # decision_values = np.where(np.isfinite(decision_values), decision_values, 0) 
        try:
            roc_score = metrics.roc_auc_score(y_test, y_test_pred_proba)
            auroc_array.append(roc_score)
        except:
            auroc_array.append(0.5)
            # print(decision_values)
        
        cm_array.append(np.array([confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]).tolist())
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        acc_array.append((tn + tp) / (tn + fp + fn + tp))
        recall_array.append(tp / (fn + tp))
        prec_array.append(tp / (fp + tp))
        spec_array.append(tn / (tn + fp))
        npv_array.append(tn / (tn + fn))
        f1sc_array.append(2 * (tp / (fn + tp)) * (tp / (fp + tp)) / ((tp / (fn + tp)) + (tp / (fp + tp))))
    
    json_dict = {
        "kernel": kernel, "C": C, "logGamma": logGamma, "degree": degree, "coef0": coef0, "n": n, "max_iter":max_iter,
        "fold Accy": acc_array,
        "avg Accy": sum(acc_array) / len(acc_array),
        "std Accy": np.std(acc_array),
        "fold Recall": recall_array,
        "avg Recall": sum(recall_array) / len(recall_array),
        "std Recall": np.std(recall_array),
        "fold Prec": prec_array,
        "avg Prec": sum(prec_array) / len(prec_array),
        "std Prec": np.std(prec_array),
        "fold Spec": spec_array,
        "avg Spec": sum(spec_array) / len(spec_array),
        "std Spec": np.std(spec_array),
        "fold Npv": npv_array,
        "avg Npv": sum(npv_array) / len(npv_array),
        "std Npv": np.std(npv_array),
        "fold F1sc": f1sc_array,
        "avg F1sc": sum(f1sc_array) / len(f1sc_array),
        "std F1sc": np.std(f1sc_array),
        "fold AUROC": auroc_array,
        "avg AUROC": sum(auroc_array) / len(auroc_array),
        "std AUROC": np.std(auroc_array),
        "confusion matrix": cm_array
    }
    
    return json_dict
        
   