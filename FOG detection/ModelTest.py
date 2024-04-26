from getDB import get_train_data
from getDB import get_tr_val_tst_data

import numpy as np
import tensorflow as tf

# Load DB
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)
X_train, X_test, y_train, y_test = get_tr_val_tst_data(all_train_data, all_features, lookback, targets)

# Create model
lstmmodel = tf.keras.models.load_model('lstmmodel.keras')

# Test model
for Id, group in all_train_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[all_features].values[0:-2],
                   df.iloc[1:][all_features].values[0:-1],
                   df.iloc[2:][all_features].values])
    X = np.reshape(X, (-1, lookback, len(all_features)))
    Y = df[targets].values[0:-2]

    scores = lstmmodel.evaluate(X, Y)
    print("Accuracy: %.2f%%" % (scores[1]*100))

scores = lstmmodel.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
