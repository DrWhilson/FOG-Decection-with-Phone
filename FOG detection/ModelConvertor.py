import tensorflow as tf
from tensorflow.keras import backend as K

print(tf.__version__)


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


lstm_model = tf.keras.models.load_model('lstm_model_test_model.keras', custom_objects={'F1_score': F1_score})

tf.saved_model.save(lstm_model, "/Models/Model")

lstm_model = tf.saved_model.load("/Models/Model")

# Convert the model.
run_model = tf.function(lambda x: lstm_model(x))

BATCH_SIZE = 1
STEPS = 10
INPUT_SIZE = 5

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], lstm_model.inputs[0].dtype))

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the model.
with open('model_LSTM.tflite', 'wb') as f:
    f.write(tflite_model)