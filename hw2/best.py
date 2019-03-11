##### ignore warnings
import warnings
warnings.filterwarnings('ignore')
##### import
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model

def sigmoid(x):                                 
    return 1 / (1 + np.exp(-x))

class binary_classification:
    def __init__(self, cross_valid):
        self.cross_valid = cross_valid

    def read_train_data(self, filename1, filename2):
        print('read train')
        self.train_X = pd.read_csv(filename1).as_matrix()
        self.train_Y = to_categorical(pd.read_csv(filename2).as_matrix())
        self.train_X = data_processing(self.train_X, Nor = True)
        print(self.train_X.shape,self.train_Y.shape)

    def read_test_data(self, filename1):
        print('read test')
        self.test_X = pd.read_csv(filename1).as_matrix()
        self.test_X = data_processing(self.test_X, Nor = False)
        print(self.test_X.shape)

    def validate(self,num):
        len3 = self.train_X.shape[0] // 10
        idx = [t for t in range(num*len3,len3 + num*len3)]
        self.valid_X = self.train_X[idx, :]
        self.train_X = np.delete(self.train_X, idx, axis = 0)
        self.valid_Y = self.train_Y[idx, :]
        self.train_Y = np.delete(self.train_Y, idx, axis = 0)

    def train(self, num):
        model = Sequential()
        model.add( Dense(input_dim = self.train_X.shape[1], units = 512, activation = 'relu'))
        model.add( Dropout(0.5) )
        for _ in range(5):
            model.add( Dense(units = 256, activation = 'relu'))
            model.add( Dropout(0.5) )
        model.add( Dense(units = 2, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()

        callbacks = []
        modelcheckpoint = ModelCheckpoint('model{}.h5'.format(num), monitor = 'val_acc', save_best_only = True)
        callbacks.append(modelcheckpoint)

        hist = model.fit(self.train_X, self.train_Y, batch_size = 128, epochs = 25, validation_data = (self.valid_X, self.valid_Y), callbacks = callbacks)

        ##### figure - loss
        
        plt.gcf().clear()
        plt.figure(figsize=(12,8))
        plt.plot(hist.history['acc'], 'b', label = 'train acc')
        plt.plot(hist.history['val_acc'], 'r', label = 'val acc')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('acc')
        plt.savefig('acc')
        
        predict_Y = model.predict(self.valid_X)
        return success_rate(np.argmax(self.valid_Y,axis=1), np.argmax(predict_Y,axis=1))
    
    def pred(self):
        model = load_model('model5.h5')
        return np.argmax(model.predict(self.test_X),axis=1)

def success_rate(real_Y, predict_Y):
    success_rate = np.sum( (real_Y - predict_Y) == 0) / real_Y.shape[0]
    return success_rate

def data_processing(X, Nor):
    ##### normalize
    if Nor:
        mean = np.mean(X, axis = 0).astype(np.float32)
        dev = np.std(X, axis = 0)
        np.save("./mean.npy", mean)
        np.save("./dev.npy", dev)
        X = (X - mean) / dev
    else:
        mean = np.load("./mean.npy")
        dev = np.load("./dev.npy")
        X = (X - mean) / dev
    ##### data processing
    # useless = []
    # X = np.delete(X, useless, axis=1)
    # age = [np.power(X[:,0], s).reshape(-1, 1) for s in [2, 3]]
    # fnlwgt = [np.power(X[:,1], s).reshape(-1, 1) for s in [2, 3]]
    # cap = [np.power(X[:,3], s).reshape(-1, 1) for s in [2, 3, 4, 5]]
    # hours = [np.power(X[:,5], s).reshape(-1, 1) for s in [2, 3]]
    # male = [np.power(X[:,76], s).reshape(-1, 1) for s in [2, 3]]
    # X = np.concatenate([X] + age + fnlwgt + cap + hours, axis = 1) 
    return X

if __name__ == '__main__':
    
    cross_valid = True
    ll = []
    
    if cross_valid:
        for i in range(10):
            model = binary_classification(True)
            model.read_train_data('X_train', 'Y_train')
            model.validate(i)
            ll.append(model.train(i))
        print(ll)
        print('Average valid loss:',np.mean(ll))
    
    else:
        model = binary_classification(False)
        model.read_train_data('X_train', 'Y_train')
        model.read_test_data('X_test')
        model.train()
        f = open('Y_gen.csv', 'w')
        f.write('id,label\n')
        for i in range(test_Y.shape[0]):
            f.write(repr(i + 1) + "," + repr(int(test_Y[i])) + "\n")
        f.close()
    
    # model = binary_classification(True)
    # model.read_test_data('X_test')
    # test_Y = model.pred()
    # f = open('Y_ker.csv', 'w')
    # f.write('id,label\n')
    # for i in range(test_Y.shape[0]):
    #     f.write(repr(i + 1) + "," + repr(int(test_Y[i])) + "\n")
    # f.close()
    