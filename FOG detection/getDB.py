import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split

import os


def get_tdcsfog_full(super_folder_path, folder_path, tdcsfog_path):
    sample_cs_path = os.path.join(super_folder_path, folder_path, tdcsfog_path)
    tdcsfog_df = []

    for dirname, _, filenames in os.walk(sample_cs_path):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))
            df['Id'] = filename[0:filename.index('.')]
            tdcsfog_df.append(df)

    tdcsfog_df = pd.concat(tdcsfog_df, ignore_index=True)
    tdcsfog_df.head()

    tdcsfog_df.StartHesitation.unique()
    tdcsfog_df.Turn.unique()
    tdcsfog_df.Walking.unique()
    return tdcsfog_df


def get_defog_full(super_folder_path, folder_path, defog_path):
    sample_cs_path = os.path.join(super_folder_path, folder_path, defog_path)
    defog_df = []

    for dirname, _, filenames in os.walk(sample_cs_path):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))
            df['Id'] = filename[0:filename.index('.')]
            defog_df.append(df)

    defog_df = pd.concat(defog_df, ignore_index=True)
    defog_df.head()

    defog_df.Valid.unique()
    defog_df.Task.unique()

    defog_df = defog_df.query('Valid==True and Task==True')
    defog_df = defog_df.drop(['Valid', 'Task'], axis=1)
    return defog_df


def get_tdcsfog_cut(super_folder_path, folder_path, tdcsfog_path):
    sample_cs_path = os.path.join(super_folder_path, folder_path, tdcsfog_path)
    tdcsfog_df = []

    for dirname, _, filenames in os.walk(sample_cs_path):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))
            df['Id'] = filename[0:filename.index('.')]
            tdcsfog_df.append(df)

    tdcsfog_df = pd.concat(tdcsfog_df, ignore_index=True)
    tdcsfog_df.head()

    return tdcsfog_df


def get_defog_cut(super_folder_path, folder_path, defog_path):
    sample_cs_path = os.path.join(super_folder_path, folder_path, defog_path)
    defog_df = []

    for dirname, _, filenames in os.walk(sample_cs_path):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))
            df['Id'] = filename[0:filename.index('.')]
            defog_df.append(df)

    defog_df = pd.concat(defog_df, ignore_index=True)
    defog_df.head()

    return defog_df


def get_train_data(targets, features, accmeasurs):
    # !Folder path
    path = r'..\DATA'
    train_path = 'train'

    # !File path
    defog_path = 'defog'
    tdcsfog_path = 'tdcsfog'
    sample = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample.head()
    sample.info()

    # !Load Train
    # Load tdcsfog_path
    tdcsfog_df = get_tdcsfog_full(path, train_path, tdcsfog_path)

    # Load defog_path
    defog_df = get_defog_full(path, train_path, defog_path)

    # Merge all train
    all_train_data = pd.concat([tdcsfog_df, defog_df])
    all_train_data = all_train_data.astype({'Time': 'int32', 'Turn': 'int8', 'Walking': 'int8',
                                            'StartHesitation': 'int8', 'AccV': 'float16',
                                            'AccML': 'float16', 'AccAP': 'float16'})
    defog_df = None
    tdcsfog_df = None

    # Delete unusual data
    all_train_data = all_train_data.loc[(all_train_data[['AccV', 'AccML', 'AccAP']] <= 9.81).all(axis=1)]
    print(all_train_data.info())

    # !Create All Feature
    all_features = [feature for feature in all_train_data.columns if
                    feature != 'Id' and feature not in targets and feature != 'Time']
    print(all_features)

    return all_features, all_train_data


def get_train_test_data(all_train_data, all_features, lookback, targets):
    # By individual person
    # for Id, group in all_train_data.groupby('Id'):
    #     df = group.set_index('Time')
    #     X = np.hstack([df[all_features].values[0:-2],
    #                    df.iloc[1:][all_features].values[0:-1],
    #                    df.iloc[2:][all_features].values])
    #     X = np.reshape(X, (-1, lookback, len(all_features)))
    #     Y = df[targets]

    # For all person
    # Chose features
    df = all_train_data.set_index('Time')
    x = np.hstack([df[all_features].values[:-2],
                   df.iloc[1:][all_features].values[:-1],
                   df.iloc[2:][all_features].values])
    x = np.reshape(x, (-1, lookback, len(all_features)))
    y = df[targets].values[:-2]

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

    return x_train, x_test, y_train, y_test


def get_test_data(targets, features, accmeasurs):
    # !Folder path
    path = r'..\DATA'
    test_path = 'test'

    # !File path
    defog_path = 'defog'
    tdcsfog_path = 'tdcsfog'
    sample = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample.head()
    sample.info()

    # !Load Test
    # Load tdcsfog_path
    tdcsfog_df = get_tdcsfog_cut(path, test_path, tdcsfog_path)

    # Load defog_path
    defog_df = get_defog_cut(path, test_path, defog_path)

    # Merge all test
    all_test_data = pd.concat([tdcsfog_df, defog_df])
    all_test_data = all_test_data.astype({'Time': 'int32',
                                          'AccV': 'float16', 'AccML': 'float16', 'AccAP': 'float16'})
    defog_df = None
    tdcsfog_df = None
    print(all_test_data.info())

    return all_test_data