from ModelDesc import LSTMModel

import numpy as np

# Load DB
from getDB import get_train_data
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']
features = ['Time', 'AccV', 'AccML', 'AccAP']
accmeasurs = ['AccV', 'AccML', 'AccAP']

all_features, all_train_data = get_train_data(targets, features, accmeasurs)

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
    Y = df[targets]

    lstmmodel.model.fit(X, Y, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

# Save model
lstmmodel.model.save('lstmmodel.keras')
