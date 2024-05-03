from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import get_tr_val_tst_data
from window_generator import WindowGenerator

import tensorflow as tf
from keras.metrics import CategoricalAccuracy, Precision
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

MAX_EPOCHS = 20


def compile_and_fit(model, window, patience=2):
    losses = [CategoricalCrossentropy()]
    metrics = [CategoricalAccuracy(), Precision()]

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

    lstm_model.model.summary()

    history = model.fit(window.train, epochs=MAX_EPOCHS,
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
wide_window = WindowGenerator(
    input_width=12000, label_width=1, shift=100,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=features)

# Create model
lstm_model = LSTMModel(all_features, lookback)

# Train model
history = compile_and_fit(lstm_model.model, wide_window)

# Test model
# lstm_model.model.evaluate(X_test, y_test)

# Save model
lstm_model.model.save('lstmmodel.keras')
