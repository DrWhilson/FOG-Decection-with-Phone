from ModelDesc import LSTMModel

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# # DB1
# from getDB_v2 import gettraindata
# lookback = 3
# feature = ['AccV', 'AccML', 'AccAP']
#
# x_train, x_val, y_train, y_val = gettraindata(feature)
#
# lstmmodel = LSTMModel(feature, lookback)
#
# lstmmodel.model.summary()
#
# batch = 5000
# epochs = 50
#
# lstmmodel.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)
#
# # Evaluation
# scores = lstmmodel.model.evaluate(x_val, y_val)
# print("Accuracy: %.2f%%" % (scores[1]*100))


# DB2
from getDB import gettraindata
lookback = 3
targets = ['StartHesitation', 'Turn', 'Walking']

all_features, all_train_data = gettraindata(targets)

print(all_features)
print(len(all_features))

lstmmodel = LSTMModel(all_features, lookback)

lstmmodel.model.summary()

batch = 5000
epochs = 50

for Id, group in all_train_data.groupby('Id'):
    df = group.set_index('Time')
    X = np.hstack([df[all_features].values[0:-2],
                   df.iloc[1:][all_features].values[0:-1],
                   df.iloc[2:][all_features].values])
    X = np.reshape(X, (-1, lookback, len(all_features)))
    Y = df[targets].values[0:-2]

    print(X.shape)
    print(Y.shape)

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=2)
    lstmmodel.model.fit(x_train, y_train, batch_size=batch, epochs=epochs, verbose=2, validation_split=.2)

    # Evaluation
    # scores = lstmmodel.model.evaluate(x_val, y_val)
    # print("Accuracy: %.2f%%" % (scores[1]*100))

    # Evaluation
    out_eval = lstmmodel.model.predict(x_val)
    eval_precision = metrics.precision_score(y_val, out_eval, average='weighted')
    eval_accuracy = metrics.accuracy_score(y_val, out_eval)
    eval_confmat = metrics.confusion_matrix(y_val, out_eval)
    print(f'the evaluation precision score is: {eval_precision}')
    print(f'the evaluation accuracy score is: {eval_accuracy}')
    print(f'the evaluation confusion matrix is : {eval_confmat}')
