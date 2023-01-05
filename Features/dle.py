from collections import Counter

import time

import numpy as np

from tqdm import tqdm

def build_vocab(data):
    word_counts = Counter(row.lower() for sample in data for row in sample)
    vocab = [w for w, f in iter(word_counts.items())]
    return vocab

def k_mers(data, n):
    kmer_array = [[s[i:i + n].lower() for i in range(len(s) - n + 1)] for s in data]
    vocab = build_vocab(kmer_array)
    return kmer_array, vocab

def linear_feature(data, k, num, nor=False):
    kmer_array, vocab = k_mers(data, k)
    features_data = []
    for s in tqdm(kmer_array):
        s = np.array(s)

        # Weights
        w = np.linspace(0, len(s) - 1, num=num)
        w = np.reshape(w, (-1, 1)) - np.arange(len(s))
        w = np.abs(w) - len(s)
        w = np.power(w , 2).T
        # 最大權重
        L = np.sum(w, axis=0)
        
        v_has = np.array([s == v for v in vocab])
        linear_data = np.dot(v_has, w) / L
        features_data.append(linear_data.tolist())
        
    features_data = np.array(features_data)
    
    if nor:
        d_max = np.reshape(np.max(features_data, axis=2), (len(features_data), -1, 1))
        d_min = np.reshape(np.min(features_data, axis=2), (len(features_data), -1, 1))
        features_data = (features_data - d_min) / (d_max - d_min)
        features_data = np.array(features_data)
    
    return features_data, vocab

def dle(seqs, k=1, power=4, normalized=0, output=0):
    """Density Linear Encoder 

    Args:
        seqs ([type]): [description]
        k (int, optional): k-mer. Defaults to 1.
        power (int, optional): max x**power. Defaults to 4.
        normalized (int, optional): 0 no normalized, 1 density, 2 seq length, 3 both. Defaults to 0.

    Returns:
        [type]: [description]
    """
    
    nor_d = True if normalized == 1 or normalized == 3 else False
    nor_l = True if normalized == 2 or normalized == 3 else False
    
    kmer_array, vocab = k_mers(seqs, k)
    
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
            
            # nor
            if nor_d and d.max() - d.min():
                d = (d - d.min()) / (d.max() - d.min())
            xlim = np.linspace(0, 1.0, num=len(d)) if nor_l else np.arange(len(d))
            
            z = np.polyfit(xlim, d, power)
            if output > 0:
                p = np.poly1d(z)
                f_array = np.array([p(np.linspace(0, 1.0 if nor_l else len(d), num=output))])
            else:
                f_array = np.array([z])
            linear_data = f_array if linear_data is None else np.append(linear_data, f_array, axis=0)
                
        features_data.append(linear_data.tolist())
        # print('Seq Len: %i | Time: %.2f s' % (len(s), time.time() - t))
    return np.array(features_data), vocab