from collections import Counter

import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

argv_dict = {"k": 1, "power": 10, "nor": 0}

for i in range(1, len(sys.argv), 2):
    argv_dict[sys.argv[i]]
    temp = int(sys.argv[i + 1])
    argv_dict[sys.argv[i]] = temp


# nor is 1 or 3 normalized density
# nor is 2 or 3 normalized seq length
nor_d = True if argv_dict["nor"] == 1 or argv_dict["nor"] == 3 else False
nor_l = True if argv_dict["nor"] == 2 or argv_dict["nor"] == 3 else False
nor_str = "_nor" if argv_dict["nor"] else ""
if nor_d:
    nor_str += "D"
if nor_l:
    nor_str += "L"

print(argv_dict)
        
def build_vocab(data):
    word_counts = Counter(row.lower() for sample in data for row in sample)
    vocab = [w for w, f in iter(word_counts.items())]
    return vocab

def k_mers(data, n):
    kmer_array = [[s[i:i + n].lower() for i in range(len(s) - n)] for s in data]
    vocab = build_vocab(kmer_array)
    return kmer_array, vocab

data = pd.read_csv("data/society/cdhit80_data_seq.csv")

kmer_array, vocab = k_mers(data['Sequence'], argv_dict["k"])

# save vocab
np.save("data/linear_features/cdhit80_k" + str(argv_dict["k"]) + "_vocab", np.array(vocab))


def features_linear_encoder(data, vocab, power=10):
    features_data = []
    for s in data:
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
        
        linear_data = np.array([])
        for v in vocab:
            d = np.where(s == v, w, 0)
            d = np.sum(d, axis=1)
            d = d / L
            
            # nor
            if nor_d and d.max() - d.min():
                d = (d - d.min()) / (d.max() - d.min())
            xlim = np.linspace(0, 1.0, num=len(d)) if nor_l else np.arange(len(d))
            
            z = np.polyfit(xlim, d, power)
            linear_data = np.append(linear_data, z)
        features_data.append(linear_data.tolist())
        print('Seq Len: %i | Time: %.2f s' % (len(s), time.time() - t))
    return features_data

features_data = features_linear_encoder(kmer_array, vocab=vocab, power=argv_dict["power"])

features_output = pd.DataFrame(features_data)
features_output["len"] = [len(s) for s in data["Sequence"]]
features_output["Cytosolic"] = data["Cytosolic"]
features_output["Nucleus"] = data["Nucleus"]
features_output.to_csv("data/linear_features/old/cdhit80_k" + str(argv_dict["k"]) + "_linear" + str(argv_dict["power"]) + nor_str +".csv", index=False)
