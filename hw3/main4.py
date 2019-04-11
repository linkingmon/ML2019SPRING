import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
import sys

class mymodel():
    def __init__(self):
        pass

    def read_train(self, filename):
        print('Start reading train data')
        data = pd.read_csv(filename).as_matrix()
        self.train_Y = to_categorical(data[:, 0], 7)
        self.train_X = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ])
        print(self.train_X.shape, self.train_Y.shape)

    def validate(self, num):
        len3 = self.train_X.shape[0] // 10
        idx = [t for t in range(num*len3,len3 + num*len3)]
        self.valid_X = self.train_X[idx, :]
        self.train_X = np.delete(self.train_X, idx, axis = 0)
        self.valid_Y = self.train_Y[idx, :]
        self.train_Y = np.delete(self.train_Y, idx, axis = 0)

    def train(self):
        num = 4
        ##### Building model
        self.model = Sequential()

        self.model.add( Conv2D(64, 3, 3, input_shape = (48, 48, 1), activation = 'relu') )
        self.model.add( BatchNormalization() )
        self.model.add( MaxPooling2D(2, 2, padding = 'same') )
        self.model.add( Dropout(0.15) )

        self.model.add( Conv2D(128, 3, 3, activation = 'relu') )
        self.model.add( BatchNormalization() )
        self.model.add( MaxPooling2D(2, 2, padding = 'same') )
        self.model.add( Dropout(0.2) )

        self.model.add( Conv2D(512, 3, 3, activation = 'relu') )
        self.model.add( BatchNormalization() )
        self.model.add( MaxPooling2D(2, 2, padding = 'same') )
        self.model.add( Dropout(0.25) )

        self.model.add( Conv2D(512, 3, 3, activation = 'relu') )
        self.model.add( BatchNormalization() )
        self.model.add( MaxPooling2D(2, 2, padding = 'same') )
        self.model.add( Dropout(0.35) )
        self.model.add( Flatten() )
        
        self.model.add( Dense(units = 512, activation = 'relu'))
        self.model.add( BatchNormalization() )
        self.model.add( Dropout(0.7) )

        self.model.add( Dense(units = 512, activation = 'relu'))
        self.model.add( BatchNormalization() )
        self.model.add( Dropout(0.7) )

        self.model.add( Dense(units = 7, activation = 'softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        ##### image processing
        train_gen = ImageDataGenerator(rotation_range = 32, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = [0.8, 1.2], horizontal_flip = True)
        train_gen.fit(self.train_X)

        ##### callbacks
        check_save = ModelCheckpoint('model4.h5',monitor='val_acc',save_best_only=True)

        ##### fit
        hist = self.model.fit_generator( train_gen.flow(self.train_X, self.train_Y, batch_size = 128), 
            steps_per_epoch =  (self.train_X.shape[0] // 128) , validation_data = (self.valid_X, self.valid_Y), epochs = 100, callbacks = [check_save])

        ##### figure - loss
        plt.gcf().clear()
        plt.figure(figsize=(12,8))
        plt.plot(hist.history['loss'], 'b', label = 'train loss')
        plt.plot(hist.history['val_loss'], 'r', label = 'val loss')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('loss')
        plt.savefig('cnnModel/model{}/loss'.format(num))

        ##### figure - acc
        plt.gcf().clear()
        plt.figure(figsize=(12,8))
        plt.plot(hist.history['acc'], 'b', label = 'train acc')
        plt.plot(hist.history['val_acc'], 'r', label = 'val acc')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('acc')
        plt.savefig('cnnModel/model{}/acc'.format(num))

if __name__ == '__main__':
    m = mymodel()
    m.read_train(sys.argv[1])
    m.validate(9)
    m.train()