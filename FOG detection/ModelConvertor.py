import tensorflow as tf
from tensorflow.keras import backend as K


def F1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


print(tf.__version__)

lstm_model = tf.keras.models.load_model('lstm_model_TEST.keras', custom_objects={'F1_score': F1_score})

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
tflite_model = converter.convert()

# Save the TFLite model
with open('lstm_model.tflite', 'wb') as f:
    f.write(tflite_model)
