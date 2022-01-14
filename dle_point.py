import time
import pickle

import numpy as np

k_array = [1, 2, 3, 4]
power_array = [8, 10]
nor_array = [2, 3]
num_array = [1, 2, 3, 5, 8, 10]

for k in k_array:
    for power in power_array:
        for nor in nor_array:
            kmer_linear_density = np.load("data/linear_features/linear/train/k" + str(k) + "p" + str(power) + "nor" + str(nor) + ".npy")
            for num in num_array:
                print("k=%s, nor=%s, num=%s" % (k, nor, num))

                data_x = None
                for row in kmer_linear_density:
                    token_data = None
                    for token_linear in row:
                        p = np.poly1d(token_linear)
                        
                        token_p = p(np.linspace(0, 1.0 , num=num))
                        token_p = np.array([token_p])
                        
                        token_data = token_p if token_data is None else np.append(token_data, token_p, axis=0)
                        
                    token_data = np.array([token_data])
                    data_x = token_data if data_x is None else np.append(data_x, token_data, axis=0)
                print(data_x.shape)
                np.save("data/linear_features/point/train/k" + str(k) + "p" + str(power) + "nor" + str(nor) + "n" + str(num), data_x)