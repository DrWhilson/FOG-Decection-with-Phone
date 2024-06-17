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

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(80, activation='tanh', input_shape=input_shape),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

    @staticmethod
    def config_run_eagerly():
        tf.config.run_functions_eagerly(True)
