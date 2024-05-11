import tensorflow as tf
from tensorflow import keras

lstm_model = tf.keras.models.load_model('lstm_model_new.keras', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('lstm_model.tflite', 'wb') as f:
    f.write(tflite_model)
