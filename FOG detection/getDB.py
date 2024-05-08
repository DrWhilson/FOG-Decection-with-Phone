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

    # Merge event types
    all_train_data['Event'] = (
                all_train_data['Turn'] | all_train_data['Walking'] | all_train_data['StartHesitation']).astype(int)

    # Drop old events type
    all_train_data = all_train_data.drop('Turn', axis=1)
    all_train_data = all_train_data.drop('Walking', axis=1)
    all_train_data = all_train_data.drop('StartHesitation', axis=1)

    # Change Event datatype to int8
    all_train_data = all_train_data.astype({'Time': 'int32', 'AccV': 'float16', 'AccML': 'float16',
                                            'AccAP': 'float16', 'Event': 'int8'})

    # Compressing time indicators
    all_train_data = all_train_data[::10]

    # Vive data describe
    print(all_train_data.info())
    print(all_train_data.head())

    # !Create All Feature
    all_features = [feature for feature in all_train_data.columns if
                    feature != 'Id' and feature not in targets and feature != 'Time']
    print(all_features)

    return all_features, all_train_data


def get_tr_val_tst_data(all_train_data, all_features, lookback, targets):
    # Create empty df for train, val, test
    train = pd.DataFrame(columns=all_train_data.columns)
    val = pd.DataFrame(columns=all_train_data.columns)
    test = pd.DataFrame(columns=all_train_data.columns)

    # Get id groups, for train 70%, val 20%, test 10%
    ids = all_train_data['Id'].unique()
    data_len = len(ids)
    train_id = ids[0:int(data_len * 0.7)]
    val_id = ids[int(data_len * 0.7):int(data_len * 0.9)]
    test_id = ids[int(data_len * 0.9):]

    # Split all_train_data to train, val, test by person id
    for Id, group in all_train_data.groupby('Id'):
        if Id in train_id:
            train = pd.concat([train, group])
        if Id in val_id:
            val = pd.concat([val, group])
        if Id in test_id:
            test = pd.concat([test, group])

    # Vive data describe
    print(train.info)
    print(val.info)
    print(test.info)

    return train, val, test


def group_split(group):
    # Get id groups, for train 70%, val 20%, test 10%
    data_len = len(group)
    train = group[0:int(data_len * 0.7)]
    val = group[int(data_len * 0.7):int(data_len * 0.9)]
    test = group[int(data_len * 0.9):]

    # Vive data describe
    print(train.info)
    print(val.info)
    print(test.info)

    return train, val, test
