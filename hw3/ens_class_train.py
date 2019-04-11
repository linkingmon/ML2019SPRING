import numpy as np
import pandas as pd
import sys
from keras.models import Sequential,load_model
from keras.utils import to_categorical
from keras.layers import Concatenate, Dense
from keras.callbacks import ModelCheckpoint

##### load ens model
m1 = load_model('model1.h5')
m2 = load_model('model2.h5')
m3 = load_model('model3.h5')
m4 = load_model('model4.h5')

#### read train data and cut validation
print('Start reading train data')
data = pd.read_csv('train.csv').as_matrix()
train_Y = to_categorical(data[:, 0], 7)
train_X = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ]) / 255
num = 9
len3 = train_X.shape[0] // 10
idx = [t for t in range(num*len3,len3 + num*len3)]
valid_X = train_X[idx, :]
train_X = np.delete(train_X, idx, axis = 0)
valid_Y = train_Y[idx, :]
train_Y = np.delete(train_Y, idx, axis = 0)

##### process train and valid data
yTest1 = m1.predict(train_X)
yTest2 = m2.predict(train_X)
yTest3 = m3.predict(train_X)
yTest4 = m4.predict(train_X)

yTest01 = m1.predict(valid_X)
yTest02 = m2.predict(valid_X)
yTest03 = m3.predict(valid_X)
yTest04 = m4.predict(valid_X)

train_concat = np.concatenate([yTest1, yTest2, yTest3, yTest4], axis = 1)
valid_concat = np.concatenate([yTest01, yTest02, yTest03, yTest04], axis = 1)


##### build ens classifier
model = Sequential()
model.add( Dense(units = 7, input_shape = (28,), activation = 'softmax') )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##### callbacks
check_save = ModelCheckpoint('model000.h5', monitor='val_acc',save_best_only=True)

##### fit
model.fit( train_concat, train_Y, batch_size = 128, validation_data = (valid_concat, valid_Y), epochs = 40, callbacks = [check_save])
model.summary()
