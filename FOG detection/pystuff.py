# Tensorflow includes
import tensorflow as tf
import keras
from keras import layers

tf.random.set_seed(7)

top_words = 5000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length: int = 32
model = keras.Sequential()
model.add(layers.Embedding(top_words, embedding_vector_length))
model.add(layers.LSTM(100))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
