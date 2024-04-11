from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_train_test_data

import numpy as np

# Load DB
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
accmeasurs = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, accmeasurs)
X_train, X_test, y_train, y_test = get_train_test_data(all_train_data, all_features, lookback, targets)

# Create model
lstmmodel = LSTMModel(all_features, lookback)

lstmmodel.model.summary()

# Train model
batch = 5000
epochs = 50

lstmmodel.model.fit(X_train, y_train, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

# Test model
lstmmodel.model.evaluate(X_test, y_test)

# Save model
lstmmodel.model.save('lstmmodel.keras')
