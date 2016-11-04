import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

# load training and test datasets
train = pd.read_csv('/Users/dueheelee/Documents/PyApp/DataMart/TubeTest/train_set.csv', parse_dates=[2,])
test = pd.read_csv('/Users/dueheelee/Documents/PyApp/DataMart/TubeTest/test_set.csv', parse_dates=[3,])

tubes = pd.read_csv('/Users/dueheelee/Documents/PyApp/DataMart/TubeTest/tube.csv')

# create some new features
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['dayofyear'] = train.quote_date.dt.dayofyear
train['dayofweek'] = train.quote_date.dt.dayofweek
train['day'] = train.quote_date.dt.day

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
test['dayofyear'] = test.quote_date.dt.dayofyear
test['dayofweek'] = test.quote_date.dt.dayofweek
test['day'] = test.quote_date.dt.day

train = pd.merge(train,tubes,on='tube_assembly_id',how='inner')
test = pd.merge(test,tubes,on='tube_assembly_id',how='inner')

train['material_id'].fillna('SP-9999',inplace=True)
test['material_id'].fillna('SP-9999',inplace=True)

# drop useless columns and create labels
idx = test.id.values.astype(int)
test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
labels = train.cost.values
train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

# convert data to numpy array
train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
    if i in [0,3,10,16,17,18,19,20,21]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

# object array to float
X_train = train.astype(float)
X_test = test.astype(float)

# train on log(1+x) for RMSLE
label_log = np.log1p(labels)

# Keras model
model = Sequential()
model.add(Dense(train.shape[1], 256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256, 256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256, 1))

model.compile(loss='mse', optimizer='rmsprop')

# train model, test on 15% hold out data
model.fit(train, label_log, batch_size=32, nb_epoch=30, verbose=2, validation_split=0.15)

# generate prediction file
preds = np.expm1(model.predict(test, verbose=0).flatten())
preds = pd.DataFrame({"id": idx, "cost": preds})
preds.to_csv('benchmark.csv', index=False)