from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_tr_val_tst_data
from window_generator import WindowGenerator

import tensorflow as tf
from keras.metrics import CategoricalAccuracy, Precision
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam


def compile_and_fit(model, window, patience=2):
    epochs = 20
    losses = [CategoricalCrossentropy()]
    metrics = [CategoricalAccuracy(), Precision()]

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

    lstm_model.model.summary()

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


# Load DB
lookback = 3
targets = ['Event']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)
train, val, test = get_tr_val_tst_data(all_train_data, all_features, lookback, targets)

# Create window
prepare_train = train.drop(['Id'], axis=1)
prepare_val = val.drop(['Id'], axis=1)
prepare_test = test.drop(['Id'], axis=1)

wide_window = WindowGenerator(
    input_width=12000, label_width=1, shift=100,
    train_df=prepare_val, val_df=prepare_test, test_df=prepare_test,
    label_columns=features)

# Create model
lstm_model = LSTMModel(wide_window, features, lookback)

# Train model
compile_and_fit(lstm_model.model, wide_window)

# Test model
# lstm_model.model.evaluate(X_test, y_test)

# Save model
lstm_model.model.save('lstm_model.keras')
