from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_tr_val_tst_data
from getDB import group_split
from window_generator import WindowGenerator

import tensorflow as tf
from keras.metrics import CategoricalAccuracy, Precision
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam


def compile_and_fit(model, window, patience=2):
    epochs = 20
    # losses = [CategoricalCrossentropy()]
    losses = ['categorical_crossentropy']
    metrics = [CategoricalAccuracy(), Precision()]

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                   patience=patience,
    #                                                   mode='min')

    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

    lstm_model.model.summary()

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val)
                        # callbacks=[early_stopping])
    return history


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

# Get characteristic window
characteristic_window = WindowGenerator(
    input_width=100, label_width=1, shift=10,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=features)

# Create model
lstm_model = LSTMModel(characteristic_window, features, lookback)

# Train model individual
for Id, group in all_train_data.groupby('Id'):
    train, val, test = group_split(group)

    individual_window = WindowGenerator(
        input_width=100, label_width=1, shift=10,
        train_df=train.drop(['Id'], axis=1),
        val_df=val.drop(['Id'], axis=1),
        test_df=test.drop(['Id'], axis=1),
        label_columns=features)

    compile_and_fit(lstm_model.model, individual_window)

# Save model
lstm_model.model.save('lstm_model.keras')
