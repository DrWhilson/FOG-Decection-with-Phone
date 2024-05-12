import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM


class LSTMModel:
    model = Sequential()

    def __init__(self, wide_window, features, batch_size=32):
        tf.config.run_functions_eagerly(True)

        input_shape = None

        for inputs, targets in wide_window.train.take(1):
            input_shape = inputs.shape[1:]

        model = Sequential(name='Prediction_Gait')
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
