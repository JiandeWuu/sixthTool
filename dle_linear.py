import time
import pickle

import numpy as np

k_array = [1, 2, 3, 4]
power_array = [4, 6, 8, 10]
nor_array = [0, 1, 2, 3]

for k in k_array:
    for power in power_array:
        for nor in nor_array:
            print("k=%s, power=%s, nor=%s" % (k, power, nor))
            
            nor_d = True if nor == 1 or nor == 3 else False
            nor_l = True if nor == 2 or nor == 3 else False
            
            with open('data/linear_features/kmer_d/k' + str(k), 'rb') as f:
                kmer_d = pickle.load(f)
                
            linear_data = None
            for seq_d in kmer_d:
                token_data = None
                for d in seq_d:
                    d = np.array(d)
                    
                    # nor
                    if nor_d and d.max() - d.min():
                        d = (d - d.min()) / (d.max() - d.min())
                    xlim = np.linspace(0, 1.0, num=len(d)) if nor_l else np.arange(len(d))
                    
                    z = np.polyfit(xlim, d, power)
                    z = np.array([z])
                    token_data = z if token_data is None else np.append(token_data, z, axis=0)
                token_data = np.array([token_data])
                linear_data = token_data if linear_data is None else np.append(linear_data, token_data, axis=0)
            print(linear_data.shape)
            np.save("data/linear_features/linear/k" + str(k) + "p" + str(power) + "nor" + str(nor), linear_data)