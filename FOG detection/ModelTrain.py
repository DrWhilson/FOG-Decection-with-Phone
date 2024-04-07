from ModelDesc import LSTMModel

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load DB
from getDB import gettraindata
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']

all_features, all_train_data = gettraindata(targets)

# Create model
lstmmodel = LSTMModel(all_features, lookback)

lstmmodel.model.summary()

# Train model
batch = 5000
epochs = 50

for Id, group in all_train_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[all_features].values[0:-2],
                   df.iloc[1:][all_features].values[0:-1],
                   df.iloc[2:][all_features].values])
    X = np.reshape(X, (-1, lookback, len(all_features)))
    Y = df[targets].values[0:-2]

    lstmmodel.model.fit(X, Y, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

# Test model


# scores = lstmmodel.model.evaluate(x_val, y_val)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# Save model
lstmmodel.model.save('lstmmodel.keras')
