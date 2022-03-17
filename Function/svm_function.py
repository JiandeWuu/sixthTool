import math

import numpy as np

def CV(x, y, fold):
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

def cv_msvm_score(x, y, folder, size=15, parameter=""):
    cv_x, cv_y = CV(x, y, folder)
    score_array = []
    for j in range(len(cv_x)):
        train_x = None
        for i in range(len(cv_x)):
            if i == j :
                test_x = cv_x[i]
                test_y = cv_y[i]
            else:
                if train_x is None:
                    train_x = cv_x[i]
                    train_y = cv_y[i]
                else:
                    train_x = np.append(train_x, cv_x[i], axis=0)
                    train_y = np.append(train_y, cv_y[i], axis=0)
        msvm = ensemble_SVM(class_weight='balanced')
        msvm.train(train_x, train_y, size=size, parameter=parameter)
        score_array.append(msvm.test(test_x, test_y))
    return score_array

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
            u_data = np.array(np.split(u_data, big_batch))
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
