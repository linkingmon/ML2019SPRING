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
            print('Raw data shape:', data.shape)
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
            print('Shape of X:', self.X.shape, '\nShape of Y:', self.Y.shape)
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
        print('Raw data shape:', data.shape)
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
                features = np.append(features, [1])
                X.append(features)
                ##### the 9th feature is PM2.5
                Y.append([concat[9, j+N]])
        self.X = np.array(X)
        self.Y = np.array(Y)
        print('Shape of X:', self.X.shape, '\nShape of Y:', self.Y.shape)

    def read_test_data(self, filename, N):
        ##### big5 encoding
        text = open(filename, 'r', encoding='big5')
        raw_data = pd.read_csv(text, header = None).as_matrix()
        ##### first 2 columns are not data
        data = raw_data[:, 2:]
        ##### check if it is only pm2.5
        if self.pm2:
            data = data[9:None:18, 9-N:].astype('float')
            self.test_X = np.array([np.append(l, [1]) for l in data])
            print('Shape of test X:', self.test_X.shape)
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
        print('Shape of test X:', self.test_X.shape)
    
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
        print('Valid Shape: ',self.X.shape, self.Y.shape, self.valid_X.shape, self.valid_Y.shape)

    def gd(self, regularize):
        iteration = 10000000
        l = 0.0001
        lamda = regularize
        lr_w = np.zeros((self.X.shape[1],1))
        ##### RMSE answer
        minc = c = np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
        print(RMSE(self.Y,np.dot(self.X, c)), RMSE(self.valid_Y,np.dot(self.valid_X, c)))
        ##### gd
        # c = np.zeros((self.X.shape[1],1))
        minn, minn_num, cnt = 2147483647, 0, 0
        for j in range(iteration):
            g = - np.dot(self.X.T, self.Y - np.dot(self.X, c) ) + 2 * lamda * c
            ##### adagrad
            lr_w += g ** 2
            c -= l * g / np.sqrt(lr_w)
            loss = RMSE(self.Y, np.dot(self.X, c))
            if self.valid_on:
                loss2 = RMSE(self.valid_Y, np.dot(self.valid_X, c))
                if loss2 < minn:
                    minn = loss2
                    minn_num = j
                    cnt = 0
                    minc = c
                else:
                    cnt += 1
                ##### early stop
                if cnt == 10:
                    break
                if j % 100 == 0:
                    print('iter{} loss: {:.5f} {:.5f}'.format(j, loss, loss2))
            self.rec_loss.append(loss)
            self.rec_loss2.append(loss2)
        if self.valid_on:
            print('Min loss: {:.5f} at iteration {}'.format(minn, minn_num))
        self.c = minc

    def plot(self):
        plt.plot(self.rec_loss[20:], 'r')
        plt.show()
        plt.plot(self.rec_loss2[20:], 'b')
        plt.show()

    def save(self):
        ##### save the model
        np.save('model{}.npy'.format(self.seed), self.c)

    def test(self, filename):
        ##### generate test_Y and save it
        test_Y = np.dot(self.test_X, self.c)
        print('Shape of test Y:',test_Y.shape)
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
            if data[i, j] < 0:
                data[i, j] = 2 * data[i, j-1] - data[i, j-2]
    return data

if __name__ == '__main__':
    ##### parameters
    n = 9
    # 2(5.7, 5.2), 8,17(5.6, 5.5), 13,34(5.6, 5.4), 21(5.7, 5.1)
    seed = 21
    onlypm2 = False
    ##### model
    model = linear_regression(seed = seed)
    if onlypm2:
        model.pm2on()
    model.read_train_data('train.csv', N = n)
    model.read_test_data('test.csv', N = n)
    # model.normal()
    model.validate()
    model.gd(regularize = 0)
    model.save()
    model.test('test_ans_random{}_hours{}_onlypm2{}.csv'.format(seed, n, onlypm2))
    # model.plot()