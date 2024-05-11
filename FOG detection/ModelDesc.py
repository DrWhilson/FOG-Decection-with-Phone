from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Precision, SparseCategoricalCrossentropy
import tensorflow as tf


class LSTMModel:
    model = tf.keras.Sequential()

    def __init__(self, wide_window, features, batch_size=32):
        tf.config.run_functions_eagerly(True)

        input_shape = None

        for inputs, targets in wide_window.train.take(1):
            input_shape = inputs.shape[1:]

        model = tf.keras.Sequential(name='Prediction_Gait')
        model.add(LSTM(80, input_shape=input_shape, return_sequences=True))

        model.add(LSTM(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(80, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(10, activation='sigmoid'))
        model.add(Dropout(0.2))

        model.add(Dense(4, activation='sigmoid'))

        self.model = model
