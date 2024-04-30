from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_tr_val_tst_data
from window_generator import WindowGenerator

import numpy as np
import tensorflow as tf


# Load DB
lookback = 3
targets = ['Event']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)
train, val, test = get_tr_val_tst_data(all_train_data, all_features, lookback, targets)

# Create window
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train, val_df=val, test_df=test,
    label_columns=features)

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
