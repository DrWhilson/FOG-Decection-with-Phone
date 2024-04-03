from ModelDesc import LSTMModel
from getDB import gettraindata

# from getDB_v2 import gettraindata

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']

all_features, all_train_data = gettraindata(targets)

lstmmodel = LSTMModel(all_features, lookback)

batch = 5000
epochs = 50
print(len(all_train_data.groupby('Id')))
for Id, group in all_train_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[all_features].values[0:-2],
                   df.iloc[1:][all_features].values[0:-1],
                   df.iloc[2:][all_features].values])
    X = np.reshape(X, (-1, lookback, len(all_features)))
    Y = df[targets].values

    lstmmodel.model.fit(X, Y, batch_size=batch, epochs=epochs, verbose=2, workers=32, validation_split=.2)
