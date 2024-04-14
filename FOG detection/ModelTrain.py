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
# X_train, X_test, y_train, y_test = get_train_test_data(all_train_data, all_features, lookback, targets)

# Create model
lstm_model = LSTMModel(all_features, lookback)

lstm_model.model.summary()

# Train model
batch = 5000
epochs = 50

for Id, group in all_train_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[all_features].values[0:-2],
                    df.iloc[1:][all_features].values[0:-1],
                    df.iloc[2:][all_features].values])
    X = np.reshape(X, (-1, lookback, len(all_features)))
    Y = df[targets]

    lstm_model.model.fit(X, Y, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

# Test model
# lstm_model.model.evaluate(X_test, y_test)

# Save model
lstm_model.model.save('lstmmodel.keras')
