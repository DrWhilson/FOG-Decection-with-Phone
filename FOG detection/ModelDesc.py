import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM


class LSTMModel(tf.keras.Model):
    def __init__(self, wide_window, features, batch_size=32):
        super(LSTMModel, self).__init__()
        self.config_run_eagerly()

        input_shape = None

        for inputs, targets in wide_window.train.take(1):
            input_shape = inputs.shape[1:]

        self.lstm_layers = [
            LSTM(80, input_shape=input_shape, return_sequences=True),
            LSTM(128, activation='relu'),
            Dropout(0.2),
            Dense(80, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(10, activation='sigmoid'),
            Dropout(0.2),
            Dense(4, activation='sigmoid')
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.lstm_layers:
            x = layer(x)
        return x

    @staticmethod
    def config_run_eagerly():
        tf.config.run_functions_eagerly(True)
