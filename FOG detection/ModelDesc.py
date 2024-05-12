import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM


class LSTMModel(tf.keras.Model):
    def __init__(self, wide_window, features, batch_size=32):
        super().__init__()
        tf.config.run_functions_eagerly(True)

        input_shape = None

        for inputs, targets in wide_window.train.take(1):
            input_shape = inputs.shape[1:]

        self.model = Sequential(name='Prediction_Gait')
        self.model.add(LSTM(80, input_shape=input_shape, return_sequences=True))

        self.model.add(LSTM(128, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))

        self.model.add(Dense(10, activation='sigmoid'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(4, activation='sigmoid'))

    def call(self, inputs):
        return self.model(inputs)
