from getDB import get_train_data
from getDB import group_split
from window_generator import WindowGenerator
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf

print(tf.__version__)

def find_median(nums):
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    return median

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

# Create model
lstm_model = tf.keras.models.load_model('lstm_model_TEST.keras', custom_objects={'F1_score': F1_score})

# Initialize constants
window_input_width = 10
window_label_width = 1
window_shift = 0
epochs = 20

print("Metrics!", lstm_model.metrics_names)

fscorelist = []

# Test model
for Id, group in all_train_data.groupby('Id'):
    train, val, test = group_split(group)

    individual_window = WindowGenerator(
            input_width=window_input_width, label_width=window_label_width, shift=window_shift,
            train_df=train.drop(['Id'], axis=1),
            val_df=val.drop(['Id'], axis=1),
            test_df=test.drop(['Id'], axis=1),
            label_columns=features)

    print("STEP!")
    _, fscore = lstm_model.evaluate(individual_window.test)
    fscorelist.append(fscore)
    print("F1Score: ", fscore)

print("===SCORE===")

for score in fscorelist:
    print("F1Score: ", score)

print("Average score:", sum(fscorelist) / len(fscorelist))
print("Median score:", find_median(fscorelist))
