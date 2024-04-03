import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def gettraindata():

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
    # print(defog)

    defog['IsFOG'] = defog[['StartHesitation', 'Walking', 'Turn']].any(axis='columns')
    print('\n', defog[['Time', 'StartHesitation', 'Walking', 'Turn', 'IsFOG']][1047890:1071070])

    print(len(defog['IsFOG'][defog['IsFOG'] == 0])+len(defog['IsFOG'][defog['IsFOG'] == 1]))


    subj_start = (defog['Time'][defog['Time'] == 0])
    subj_start_ind = np.array(subj_start.index)
    print(len(subj_start_ind))

    subj_end_ind = subj_start_ind[1:] - 1
    print(len(subj_end_ind))

    print('FOG event at head of subject number: ', np.where(defog['IsFOG'][subj_start_ind] == 1))

    print('FOG event at tail of subject number: ', np.where(defog['IsFOG'][subj_end_ind] == 1))

    x = defog[['AccV', 'AccML', 'AccAP']]
    y = defog['IsFOG']

    X_train, X_test, Y_train, Y_test_defog = train_test_split(x, y, test_size=0.1, random_state=1)

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    fog_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'verbose': 1,
        'max_depth': 6,
        'num_leaves': 50
    }

    return x_train, x_val, y_train, y_val

