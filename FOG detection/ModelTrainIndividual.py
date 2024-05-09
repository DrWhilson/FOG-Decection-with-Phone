from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_tr_val_tst_data
from getDB import group_split
from window_generator import WindowGenerator

import tensorflow as tf
from keras.metrics import CategoricalAccuracy, Precision
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam


# Load DB
lookback = 3
targets = ['Event']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)

# Пet the first patient's data
ids = all_train_data['Id'].unique()
characteristic_group = all_train_data[all_train_data['Id'] == ids[0]]

# Split first patient's data
train, val, test = group_split(characteristic_group)

# Initialize constants
window_input_width = 10
window_label_width = 1
window_shift = 0
epochs = 20
losses = ['binary_crossentropy']
metrics = [CategoricalAccuracy(), Precision()]

# Get characteristic window
characteristic_window = WindowGenerator(
    input_width=window_input_width, label_width=window_label_width, shift=window_shift,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=features)

# Create model
lstm_model = LSTMModel(characteristic_window, features, lookback)
lstm_model.model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)
lstm_model.model.summary()

# Train model individual
for Id, group in all_train_data.groupby('Id'):
    train, val, test = group_split(group)

    print("!Len: ", len(train))
    print("!Events:", train['Event'].value_counts())

    individual_window = WindowGenerator(
        input_width=window_input_width, label_width=window_label_width, shift=window_shift,
        train_df=train.drop(['Id'], axis=1),
        val_df=val.drop(['Id'], axis=1),
        test_df=test.drop(['Id'], axis=1),
        label_columns=features)

    lstm_model.model.fit(individual_window.train, epochs=epochs,
                         validation_data=individual_window.val)

# Save model
lstm_model.model.save('lstm_model.keras')
