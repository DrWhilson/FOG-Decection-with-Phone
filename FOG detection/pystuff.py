# Tensorflow includes
import tensorflow as tf
import keras
from keras import layers

tf.random.set_seed(7)
top_words = 5000
embedding_vector_length: int = 32


class MyNet:
    model = keras.Sequential()
    def __init__(self):
        super(MyNet, self).__init__()

        self.model.add(layers.Embedding(top_words, embedding_vector_length))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.LSTM(100))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(self.model.summary())

    def forward(self, x):
        return self.model


def prepare_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)

    max_review_length = 500
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)

    return (X_train, y_train), (X_test, y_test)


def train(model, X_test, X_train, y_test, y_train):
    ret = {}
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=7, batch_size=64)

    scores = model.evaluate(X_test, y_test, verbose=0)
    ret['accuracy'] = scores[1] * 100

    if ret:
        return ret['accuracy']


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = prepare_data()
    net = MyNet()
    accuracy = train(net.model, X_test, X_train, y_test, y_train)

    print("Accuracy: %.2f%%" % accuracy)
