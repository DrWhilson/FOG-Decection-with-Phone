from getDB import get_train_data
from getDB import get_train_test_data

import numpy as np
import tensorflow as tf

# Load DB
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)
X_train, X_test, y_train, y_test = get_train_test_data(all_train_data, all_features, lookback, targets)

# Create model
lstmmodel = tf.keras.models.load_model('lstmmodel.keras')

# Test model
scores = lstmmodel.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
