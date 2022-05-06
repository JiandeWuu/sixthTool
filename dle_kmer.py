import time 
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

def build_vocab(data):
    word_counts = Counter(row.lower() for sample in data for row in sample)
    vocab = [w for w, f in iter(word_counts.items())]
    return vocab

def k_mers(data, n):
    kmer_array = [[s[i:i + n].lower() for i in range(len(s) - n)] for s in data]
    vocab = build_vocab(kmer_array)
    return kmer_array, vocab

data = pd.read_csv("data/society/cdhit80_0320_loc75.csv")

k_array = [1, 2, 3, 4]

for k in k_array:
    kmer_array, vocab = k_mers(data['Sequence'], k)
        
    features_data = []
    for s in kmer_array:
        s = np.array(s)
        t = time.time()
        # Weights
        w = np.reshape(np.arange(len(s)), (-1, 1)) - np.arange(len(s))
        w_abs = np.abs(w)
        w_max = (w.max() + 1)
        w = w_abs - w_max
        w = np.power(w , 2)
        # 最大權重
        L = np.sum(w, axis=0)
        
        linear_data = None
        for v in vocab:
            d = np.where(s == v, w, 0)
            d = np.sum(d, axis=1)
            d = d / L
            f_array = np.array([d])
            linear_data = f_array if linear_data is None else np.append(linear_data, f_array, axis=0)
        linear_data = np.array(linear_data)
        features_data.append(linear_data.tolist())
    
    with open("data/linear_features/kmer_d/0320/k" + str(k), 'wb') as f:
        pickle.dump(features_data, f)