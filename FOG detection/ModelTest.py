from getDB import get_train_data
from getDB import group_split
from window_generator import WindowGenerator

import numpy as np
import tensorflow as tf

# Load DB
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)

# Create model
lstm_model = tf.keras.models.load_model('lstm_model.keras')

# Initialize constants
window_input_width = 10
window_label_width = 1
window_shift = 0
epochs = 20

# Test model
for Id, group in all_train_data.groupby('Id'):
    train, val, test = group_split(group)

    individual_window = WindowGenerator(
        input_width=window_input_width, label_width=window_label_width, shift=window_shift,
        train_df=train.drop(['Id'], axis=1),
        val_df=val.drop(['Id'], axis=1),
        test_df=test.drop(['Id'], axis=1),
        label_columns=features)

    print("STEP!")
    val_performance = lstm_model.evaluate(individual_window.val, return_dict=True)
    performance = lstm_model.evaluate(individual_window.test, verbose=0, return_dict=True)
