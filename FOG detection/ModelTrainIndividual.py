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

# Получаем данные первого пациента
ids = all_train_data['Id'].unique()
characteristic_group = all_train_data[all_train_data['Id'] == ids[0]]

# Разделяем данные на тренеровачные, тестовые и валидационные
train, val, test = group_split(characteristic_group)

# Создаём константы
window_input_width = 10
window_label_width = 1
window_shift = 0
epochs = 20
losses = ['binary_crossentropy']
metrics = [F1_score, 'precision', 'accuracy']

# Создаём окно данных
characteristic_window = WindowGenerator(
    input_width=window_input_width, label_width=window_label_width, shift=window_shift,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=targets)

# Создаём модель
lstm_model = LSTMModel(characteristic_window, features, lookback)
lstm_model.model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)
lstm_model.model.summary()

# Обучаем модель
for Id, group in all_train_data.groupby('Id'):
    # Разделяем данные на тренеровачные, тестовые и валидационные
    train, val, test = group_split(group)

    if len(train['Event'].unique()) == 1:
        continue

    print("!Len: ", len(train))
    print("!Events:", train['Event'].value_counts())

    # Создаём окно данных
    individual_window = WindowGenerator(
        input_width=window_input_width, label_width=window_label_width, shift=window_shift,
        train_df=train.drop(['Id'], axis=1),
        val_df=val.drop(['Id'], axis=1),
        test_df=test.drop(['Id'], axis=1),
        label_columns=targets)

    # Дообучаем модель
    lstm_model.model.fit(individual_window.train, epochs=epochs,
                         validation_data=individual_window.val)
    break

# Save model
lstm_model.model.save('lstm_least.keras')

# Convert the model.
run_model = tf.function(lambda x: lstm_model.model(x))
# This is important, let's fix the input size.
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
with open('lite_lstm_least.tflite', 'wb') as f:
    f.write(tflite_model)
