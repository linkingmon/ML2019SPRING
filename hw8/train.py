import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
import sys
from mobilenet3 import MobileNet

filename = sys.argv[1]
data = pd.read_csv(filename).as_matrix()
train_Y = to_categorical(data[:, 0], 7)
train_X = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1]]) / 255

sp = len(train_X) // 10
valid_X = train_X[:sp]
valid_Y = train_Y[:sp]
train_X = train_X[sp:]
train_Y = train_Y[sp:]
print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)

model = MobileNet()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

train_gen = ImageDataGenerator(rotation_range=32, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2], horizontal_flip=True)
train_gen.fit(train_X)

csv_logger = CSVLogger('training.log')
check_save = ModelCheckpoint('model3.h5', save_weights_only=True, monitor='val_acc', save_best_only=True)

model.fit_generator(train_gen.flow(train_X, train_Y, batch_size=64),
                    steps_per_epoch=(train_X.shape[0] // 64), validation_data=(valid_X, valid_Y), epochs=500, callbacks=[check_save, csv_logger])
