import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import optimizers
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
import keras.backend as K
##### ignore warnings
import warnings
warnings.filterwarnings('ignore')
##### import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class linear_regression:
    def __init__(self, seed):
        self.rec_loss = []
        self.rec_loss2 = []
        self.valid_on = False
        self.seed = seed
        self.pm2 = False

    def read_train_data(self, filename, N):
        ##### big5 encoding
        text = open(filename, 'r', encoding = 'big5')
        raw_data = pd.read_csv(text).as_matrix()
        ##### first 3 columns are not data
        data = raw_data[:, 3:]
        ##### check if it is only pm2.5
        if self.pm2:
            data = np.vsplit(data_processing(data[9:None:18, :].astype('float')), 20)
            data = np.concatenate(data, axis = 1)
            # print('Raw data shape:', data.shape)
            X, Y = [], []
            for i in range(data.shape[0]):
                for j in range(0, data.shape[1] - N):
                    ##### the data of the previous N hours
                    ##### adding bias
                    features = data[i, j:j+N]
                    X.append(features)
                    Y.append([data[i, j+N]])
            self.X = np.array(X)
            self.Y = np.array(Y)
            # print('Shape of X:', self.X.shape, '\nShape of Y:', self.Y.shape)
            return
        ##### replace 'NR' with 0
        del_list = []
        for i in range(10, len(data), 18):
            del_list.append(i)
        for idx in del_list:
            for j in range(data.shape[1]):
                if data[idx, j] == 'NR':
                    data[idx, j] = '0'
        ##### change str to float
        data = data_processing(data.astype('float'))
        # print('Raw data shape:', data.shape)
        ##### 18-features for one day, 20 days per month
        X, Y = [], []
        for i in range(0, data.shape[0], 18*20):
            ##### shape 20 * (18, 24)
            days = np.vsplit(data[i:i+18*20], 20)
            ##### concat sane features
            concat = np.concatenate(days, axis = 1)
            for j in range(0, concat.shape[1] - N):
                ##### the data of the previous N hours
                features = concat[:, j:j+N].flatten()
                ##### adding bias
                X.append(features)
                ##### the 9th feature is PM2.5
                Y.append([concat[9, j+N]])
        self.X = np.array(X)
        self.Y = np.array(Y)
        # print('Shape of X:', self.X.shape, '\nShape of Y:', self.Y.shape)

    def read_test_data(self, filename, N):
        ##### big5 encoding
        text = open(filename, 'r', encoding='big5')
        raw_data = pd.read_csv(text, header = None).as_matrix()
        ##### first 2 columns are not data
        data = raw_data[:, 2:]
        ##### check if it is only pm2.5
        if self.pm2:
            self.test_X = data_processing(data[9:None:18, 9-N:].astype('float'))
            # print('Shape of test X:', self.test_X.shape)
            # print(self.test_X)
            return
        ##### replace 'NR' with 0
        del_list = []
        for i in range(10, len(data), 18):
            del_list.append(i)
        for idx in del_list:
            for j in range(data.shape[1]):
                if data[idx, j] == 'NR':
                    data[idx, j] = 0
        ##### shange str to float
        data = data_processing(data.astype('float'))
        ##### 18-features for one day, 20 days per month
        X = []
        for i in range(0, data.shape[0], 18):
            days = data[i:i+18,(9-N):9]
            days = days.flatten()
            X.append(days)
        self.test_X = np.array(X)
        # print('Shape of test X:', self.test_X.shape)
    
    def pm2on(self):
        self.pm2 = True
        
    def normal(self):
        ##### normalize both train ans test data (with same mu ans sig)
        for i in range(self.X.shape[1]-1):
            mu = np.mean(self.X[:, i], axis = 0)
            sig = np.std(self.X[:, i], axis = 0)
            self.X[:, i] = (self.X[:, i] - mu) / sig
            self.test_X[:, i] = (self.test_X[:, i] - mu) / sig

    def validate(self):
        ##### random cutting validation set: 9/10 + 1/10
        self.valid_on = True
        lenn = self.X.shape[0] // 10
        np.random.seed(self.seed)
        idx = np.random.randint(self.X.shape[0], size = lenn)
        self.valid_X = self.X[idx, :]
        self.X = np.delete(self.X, idx, axis = 0)
        self.valid_Y = self.Y[idx, :]
        self.Y = np.delete(self.Y, idx, axis = 0)
        # print('Valid Shape: ',self.X.shape, self.Y.shape, self.valid_X.shape, self.valid_Y.shape)

    def train(self):
        model = Sequential()
        model.add( Dense(input_dim = self.X.shape[1], units = 10, activation = 'relu'))
        model.add( Dense(units = 10))
        model.add( Dense(units = 10))
        model.add( Dense(units = 1))
        model.compile(loss='mean_squared_error', optimizer = 'adam')
        model.summary()
        hist = model.fit(self.X, self.Y, batch_size = 128, epochs = 50, validation_data = (self.valid_X, self.valid_Y))

        ##### figure - loss
        plt.gcf().clear()
        plt.figure(figsize=(12,8))
        plt.plot(hist.history['loss'], 'b', label = 'train loss')
        plt.plot(hist.history['val_loss'], 'r', label = 'val loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.savefig('loss')

    def save(self):
        ##### save the model
        np.save('model{}.npy'.format(self.seed), self.c)

    def test(self, filename):
        ##### generate test_Y and save it
        test_Y = np.dot(self.test_X, self.c)
        # print('Shape of test Y:',test_Y.shape)
        with open(filename, 'w') as f:
            f.write('id,value\n')
            for i in range(test_Y.shape[0]):
                f.write('id_{},{}\n'.format(i,test_Y[i][0]))
            f.close()

def RMSE(Y_real, Y_predict):
    loss = np.sqrt(np.linalg.norm(Y_real - Y_predict) ** 2 / Y_real.shape[0])
    return loss

def rms(Y_real, Y_predict):
    loss = K.sqrt(K.linalg.norm(Y_real - Y_predict) ** 2 / Y_real.shape[0])
    return loss

def data_processing(data):
    for i in range(data.shape[0]):
        for j in range(2, data.shape[1]):
            if data[i, j] <= 0:
                data[i, j] = np.nan
    for i in range(data.shape[0]):
        s = pd.Series(data[i])
        data[i] = (s.interpolate().as_matrix())
    return data

if __name__ == '__main__':
    ##### parameters
    n = 9
    seed = 1
    onlypm2 = False
    ##### model
    model = linear_regression(seed = seed)
    if onlypm2:
        model.pm2on()
    model.read_train_data('train.csv', N = n)
    model.read_test_data('test.csv', N = n)
    # model.normal()
    model.validate()
    model.train()
    # model.save()
    # model.test('test_ans_random{}_hours{}_onlypm2{}.csv'.format(seed, n, onlypm2))