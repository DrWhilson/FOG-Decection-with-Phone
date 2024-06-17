from getDB import get_train_data
from getDB import group_split
from window_generator import WindowGenerator

import tensorflow as tf

print(tf.__version__)

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
metrics = ['precision', 'accuracy']

# Get characteristic window
characteristic_window = WindowGenerator(
    input_width=window_input_width, label_width=window_label_width, shift=window_shift,
    train_df=train.drop(['Id'], axis=1),
    val_df=val.drop(['Id'], axis=1),
    test_df=test.drop(['Id'], axis=1),
    label_columns=targets)

print("Create Window!")

# Create model
input_shape = None

for inputs, _ in characteristic_window.train.take(1):
    input_shape = inputs.shape[1:]

print("Problem shape! ", input_shape)

lstm_model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=input_shape, name='input'),
                            tf.keras.layers.LSTM(80, activation='tanh', return_sequences=True),
                            tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False),
                            tf.keras.layers.Dense(80, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dropout(0.9),
                            tf.keras.layers.Dense(10, activation='sigmoid'),
                            tf.keras.layers.Dropout(0.9),
                            tf.keras.layers.Dense(1, activation='sigmoid')
                            ])

lstm_model.compile(loss=losses, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)
print("Create Model")

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
        label_columns=targets)

    lstm_model.fit(individual_window.train, epochs=epochs,
                   validation_data=individual_window.val)

    lstm_model.evaluate(individual_window.test)

return
run_model = tf.function(lambda x: lstm_model(x))
# Это важно. Поправляем размерности
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

# Сохраняем конвертированную модель
with open('lstm_model.tflite', 'wb') as f:
    f.write(tflite_model)
