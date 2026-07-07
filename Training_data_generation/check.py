import os
import numpy as np

id = '0724'


# os.system(f'cd Training_data{id}')

x = np.load(f'Training_data{id}/homogenize.npz', allow_pickle=True)

print(len(list(x.keys())))
