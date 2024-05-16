from ModelDesc import LSTMModel
from getDB import get_train_data
from getDB import group_split
from window_generator import WindowGenerator

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


# Load DB
lookback = 3
targets = ['Event']
features = ['Time', 'AccV', 'AccML', 'AccAP']
acc_measures = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, acc_measures)

# Пet the first patient's data
ids = all_train_data['Id'].unique()
characteristic_group = all_train_data[all_train_data['Id'] == ids[0]]

# Split first patient's data
train, val, test = group_split(characteristic_group)

# Initialize constants
window_input_width = 10
window_label_width = 1
window_shift = 0
epochs = 20
losses = ['binary_crossentropy']
metrics = [F1_score]

# Get characteristic window
characteristic_window = WindowGenerator(
    input_width=window_input_width, label_width=window_label_width, shift=window_shift,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=features)

# Create model
lstm_model = LSTMModel(characteristic_window, features, lookback)
lstm_model.model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)
lstm_model.model.summary()

# Train model individual
for Id, group in all_train_data.groupby('Id'):
    train, val, test = group_split(group)

    print("!Len: ", len(train))
    print("!Events:", train['Event'].value_counts())

    individual_window = WindowGenerator(
        input_width=window_input_width, label_width=window_label_width, shift=window_shift,
        train_df=train.drop(['Id'], axis=1),
        val_df=val.drop(['Id'], axis=1),
        test_df=test.drop(['Id'], axis=1),
        label_columns=features)

    lstm_model.model.fit(individual_window.train, epochs=epochs,
                         validation_data=individual_window.val)

# Save model
lstm_model.model.save('lstm_model_least.keras')

# Convert the model.
run_model = tf.function(lambda x: lstm_model.model(x))

BATCH_SIZE = 1
STEPS = 10
INPUT_SIZE = 5

concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], lstm_model.model.inputs[0].dtype))

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model.model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True

tflite_model = converter.convert()

# Save the model.
with open('lite_lstm_model_least.tflite', 'wb') as f:
    f.write(tflite_model)
