import numpy as np
import tensorflow as tf

# Load DB
from getDB import get_test_data

lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
accmeasurs = ['AccV', 'AccML', 'AccAP']

all_test_data = get_test_data(targets, features, accmeasurs)

# Create model
lstmmodel = tf.keras.models.load_model('lstmmodel.keras')

# Test model
for Id, group in all_test_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[accmeasurs].values[0:-2],
                   df.iloc[1:][accmeasurs].values[0:-1],
                   df.iloc[2:][accmeasurs].values])
    X = np.reshape(X, (-1, lookback, len(accmeasurs)))
    print(df.shape)
    Y = df[targets].values[0:-2]

    scores = lstmmodel.model.evaluate(X, Y)
    print("Accuracy: %.2f%%" % (scores[1]*100))
