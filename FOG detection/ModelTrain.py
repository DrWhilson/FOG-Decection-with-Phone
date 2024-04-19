from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_train_test_data

import numpy as np

# Load DB
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)
X_train, X_test, y_train, y_test = get_train_test_data(all_train_data, all_features, lookback, targets)

# Create model
lstm_model = LSTMModel(all_features, lookback)

lstm_model.model.summary()

# Train model
batch = 5000
epochs = 50

lstm_model.model.fit(X_train, y_train, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

# Test model
lstm_model.model.evaluate(X_test, y_test)

# Save model
lstm_model.model.save('lstmmodel.keras')
