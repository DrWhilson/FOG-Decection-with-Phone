from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, LSTM
from keras.optimizers import Adam, SGD
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Precision, SparseCategoricalCrossentropy
import tensorflow as tf


class LSTMModel:
    model = Sequential()

    def __init__(self, all_features, lookback):
        tf.config.run_functions_eagerly(True)

        # losses = ['categorical_crossentropy']
        # metrics = [CategoricalAccuracy(), Precision()]

        model = Sequential(name='Prediction_Gait')
        model.add(LSTM(80, input_shape=(lookback, len(all_features),), return_sequences=True))

        model.add(LSTM(128, activation='relu'))

        model.add(Dense(80, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(10, activation='sigmoid'))

        model.add(Dense(3, activation='softmax'))
        # model.compile(optimizer=Adam(learning_rate=0.01), loss=losses, metrics=metrics)

        self.model = model
