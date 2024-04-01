import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from math import sqrt

DATA_ROOT_DEFOG = r'../DATA/train/defog/'
defog = pd.DataFrame()
for root, dirs, files in os.walk(DATA_ROOT_DEFOG):
    for name in files:
        f = os.path.join(root, name)
        df_list = pd.read_csv(f)
        words = name.split('.')[0]
        df_list['file'] = name.split('.')[0]
        defog = pd.concat([defog, df_list], axis=0)

keys = np.arange(len(defog))
defog = defog.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=True)
print(defog)

