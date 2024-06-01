from getDB import get_train_data
from getDB import group_split
from window_generator import WindowGenerator

import tensorflow as tf
from tensorflow.keras import Metric
import tensorflow.keras as keras
from tensorflow.keras import backend as K


class F1_score(Metric):
    def __init__(self, name='F1_score', **kwargs):
        super(F1_score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=self.dtype))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), dtype=self.dtype))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), dtype=self.dtype))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def get_config(self):
        base_config = super(F1_score, self).get_config()
        return base_config

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)


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
metrics = [F1_score, 'accuracy', 'precision']

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

lstm_model = tf.keras.Sequential([
                            tf.keras.layers.LSTM(80, activation='tanh', input_shape=input_shape, return_sequences=True),
                            tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False),
                            tf.keras.layers.Dense(80, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(10, activation='sigmoid'),
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
    break
