import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal

import os


def get_tdcsfog(super_folder_path, folder_path, tdcsfog_path):
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

def get_defog(super_folder_path, folder_path, defog_path):
    sample_cs_path = os.path.join(super_folder_path, folder_path, defog_path)
    defog_df = []

    for dirname, _, filenames in os.walk(sample_cs_path):
        for filename in filenames:
            df = pd.read_csv(os.path.join(dirname, filename))
            df['Id'] = filename[0:filename.index('.')]
            defog_df.append(df)

    defog_df = pd.concat(defog_df, ignore_index=True)
    defog_df.head()

    features = ['Time', 'AccV', 'AccML', 'AccAP']
    accmeasurs = ['AccV', 'AccML', 'AccAP']
    defog_df.Valid.unique()
    defog_df.Task.unique()

    defog_df = defog_df.query('Valid==True and Task==True')
    defog_df = defog_df.drop(['Valid', 'Task'], axis=1)
    return defog_df
def gettraindata(targets):
    # Folder path
    path = r'..\DATA'
    train_path = 'train'
    test_path = 'test'

    # File path
    defog_path = 'defog'
    tdcsfog_path = 'tdcsfog'
    sample = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    sample.head()
    sample.info()

    # !Load Train
    # Load tdcsfog_path
    tdcsfog_df = get_tdcsfog(path, train_path, tdcsfog_path)

    # Load defog_path
    defog_df = get_defog(path, train_path, defog_path)

    # Merge all train
    all_train_data = pd.concat([tdcsfog_df, defog_df])
    all_train_data = all_train_data.astype({'Time': 'int32', 'Turn': 'int8', 'Walking': 'int8',
                                            'StartHesitation': 'int8', 'AccV': 'float16',
                                            'AccML': 'float16', 'AccAP': 'float16'})
    defog_df = None
    tdcsfog_df = None
    all_train_data.info()

    # Create All Feature
    all_features = [feature for feature in all_train_data.columns if
                    feature != 'Id' and feature not in targets and feature != 'Time']

    return all_features, all_train_data
