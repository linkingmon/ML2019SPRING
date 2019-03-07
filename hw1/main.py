##### ignore warnings
import warnings
warnings.filterwarnings('ignore')
##### import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class linear_regression:
    def __init__(self, cross_valid):
        self.pm2 = False
        self.cross_valid = cross_valid

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
                    features = np.append(data[i, j:j+N], [1])
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
        data = data.astype('float')
        # print('Raw data shape:', data.shape)
        ##### 18-features for one day, 20 days per month
        X, Y = [], []
        for i in range(0, data.shape[0], 18*20):
            ##### shape 20 * (18, 24)
            days = np.vsplit(data[i:i+18*20], 20)
            ##### concat sane features
            concat = data_processing(np.concatenate(days, axis = 1))
            for j in range(0, concat.shape[1] - N):
                ##### the data of the previous N hours
                features = concat[:, j:j+N].flatten()
                ##### adding bias
                features = np.append(features, [1])
                X.append(features)
                ##### the 9th feature is PM2.5
                Y.append([concat[9, j+N]])
        self.X = np.array(X)
        self.Y = np.array(Y)

    def read_test_data(self, filename, N):
        ##### big5 encoding
        text = open(filename, 'r', encoding = 'big5')
        raw_data = pd.read_csv(text, header = None).as_matrix()
        ##### first 2 columns are not data
        data = raw_data[:, 2:]
        ##### check if it is only pm2.5
        if self.pm2:
            data = data[9:None:18, 9-N:].astype('float')
            self.test_X = np.array([np.append(l, [1]) for l in data])
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
        data = data.astype('float')
        ##### 18-features for one day, 20 days per month
        X = []
        for i in range(0, data.shape[0], 18):
            days = data[i:i+18,(9-N):9]
            days = days.flatten()
            days = np.append(days, [1])
            X.append(days)
        self.test_X = np.array(X)

    def shape(self):
        print('Shape of X:', self.X.shape, '\nShape of Y:', self.Y.shape)
        if self.cross_valid:
            print('Shape of valid_X:', self.valid_X.shape, '\nShape of valid_Y:', self.valid_Y.shape)
        else:
            print('Shape of test_X:', self.test_X.shape)


    
    def pm2on(self):
        self.pm2 = True

    def validate(self,num):
        len3 = self.X.shape[0] // 10
        idx = [t for t in range(num*len3,len3 + num*len3)]
        self.valid_X = self.X[idx, :]
        self.X = np.delete(self.X, idx, axis = 0)
        self.valid_Y = self.Y[idx, :]
        self.Y = np.delete(self.Y, idx, axis = 0)

    def gd(self, regularize):
        ##### RMSE answer
        iteration = 50000
        l = 0.001
        lamda = regularize
        lr_w = np.zeros((self.X.shape[1],1))
        ##### RMSE answer
        c = np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
        ##### gd
        '''
        minn = 2147483647
        for j in range(iteration):
            g = - np.dot(self.X.T, self.Y - np.dot(self.X, c) ) + 2 * lamda * c
            ##### adagrad
            lr_w += g ** 2
            c -= l * g / np.sqrt(lr_w)
            loss = RMSE(self.Y, np.dot(self.X, c))
            if loss < minn:
                if j % 100 == 0:
                    pass
                    print('iter{} loss: {}'.format(j, loss))
                minn = loss
            else:
                break
        '''
        self.c = c
        ##### results
        if self.cross_valid:
            print('Train loss: {}'.format(RMSE(self.Y, np.dot(self.X, self.c))))
            print('Valid loss: {}'.format(RMSE(self.valid_Y, np.dot(self.valid_X, self.c))))
            return RMSE(self.valid_Y, np.dot(self.valid_X, self.c))
        else:
            print('Train loss: {}'.format(RMSE(self.Y, np.dot(self.X, self.c))))            

    def save(self):
        ##### save the model
        np.save('model.npy', self.c)

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

def data_processing(data):
    for i in range(data.shape[0]):
        for j in range(2, data.shape[1]):
            if data[i, j] <= 0 and (i == 9 or i == 8):
                data[i, j] = np.nan
    for i in range(data.shape[0]):
        s = pd.Series(data[i])
        data[i] = (s.interpolate(limit_direction = 'both').as_matrix())
    return data

if __name__ == '__main__':
    ##### parameters
    n = 9
    onlypm2 = False
    cross_valid = False
    lamda = 0
    ##### model
    ll = []
    if cross_valid:
        for i in range(10):
            model = linear_regression(True)
            model.read_train_data('train.csv', N = n)
            model.validate(i)
            model.shape()
            ll.append(model.gd(regularize = lamda))
        print('Average valid loss:',np.mean(ll))
    else:
        model = linear_regression(False)
        if onlypm2:
            model.pm2on()
        model.read_train_data('train.csv', N = n)
        model.read_test_data('test.csv', N = n)
        model.shape()
        model.gd(regularize = lamda)
        model.save()
        model.test('test_ans.csv')
