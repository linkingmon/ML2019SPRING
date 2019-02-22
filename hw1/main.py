##### ignore warnings
import warnings
warnings.filterwarnings('ignore')
##### import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

def read_train_data(filename, N):
    # print('train start')
    ##### big5 encoding
    text = open(filename, 'r', encoding='big5')
    raw_data = pd.read_csv(text).as_matrix()
    # print(raw_data[0], '\n')
    ##### first 3 columns are not data
    data = raw_data[:, 3:]
    # print(data)
    # print(type(data))
    # print(type(data[0][0]))
    ##### delete 'NR'
    del_list = []
    for i in range(10, len(data), 18):
        del_list.append(i)
    # print('del_list', del_list)
    data = np.delete(data, del_list, axis = 0)
    # print(data)
    ##### shange str to float
    data = data.astype('float')
    # print(data)
    # print(type(data))
    # print(type(data[0][0]))
    ##### 17-features for one day, 20 days per month
    X, Y = [], []
    for i in range(0, data.shape[0], 17*20):
        ##### shape 20 * (17, 24)
        days = np.vsplit(data[i:i+17*20], 20)
        # print(type(days))
        # print(days.shape)
        # print(days)
        ##### concat sane features
        concat = np.concatenate(days, axis = 1)
        # print(concat)
        # print(concat.shape)
        for j in range(0, concat.shape[1] - N):
            ##### the data of the previous N hours
            features = concat[:, j:j+N].flatten()
            ##### add w0
            features = np.append(features, [1])
            # print(features)
            X.append(features)
            ##### the 9th feature is PM2.5
            # print("Y", concat[9, j+N])
            Y.append([concat[9, j+N]])
    X = np.array(X)
    Y = np.array(Y)
    print('Shape of X:', X.shape, '\nShape of Y:', Y.shape)
    return X, Y

def read_test_data(filename, N):
    # print('test start')
    ##### big5 encoding
    text = open(filename, 'r', encoding='big5')
    raw_data = pd.read_csv(text, header = None).as_matrix()
    ##### first 2 columns are not data
    data = raw_data[:, 2:]
    ##### delete 'NR'
    del_list = []
    for i in range(10, len(data), 18):
        del_list.append(i)
    data = np.delete(data, del_list, axis = 0)
    ##### shange str to float
    data = data.astype('float')
    ##### 17-features for one day, 20 days per month
    X = []
    for i in range(0, data.shape[0], 17):
        ##### shape 20 * (17, 24)
        days = data[i:i+17,(9-N):9]
        days = days.flatten()
        days = np.append(days, [1])
        # print(days)
        X.append(days)
    X = np.array(X)
    # print('Shape of X:', X.shape)
    return X

def grad_descent(X, Y, plot_on):
    ##### Y = X * W
    ##### grad = - 2 * W.T * (Y - X * W) / m
    W = np.ones((X.shape[1], 1))
    ##### ada_grad
    lr_w = np.zeros((X.shape[1], 1))
    l_rate = 1
    iteration = 10000000
    loss_prev = 0
    thres = 0.0001
    ##### recording loss
    rec_loss = []
    for i in range(iteration):
        # print(X.shape, Y.shape, W.shape)
        grad = - 2 * np.dot( X.T, ( Y - np.dot(X,W) ) ) / Y.shape[0]
        lr_w = lr_w + grad ** 2
        W = W - l_rate * grad / np.sqrt(lr_w)
        if(i % 10 == 0):
            print('iteration{}: '.format(i), end = '')
            loss = MSE(Y, np.dot(X,W))
            if( np.abs(loss_prev - loss) <= thres):
                break
            loss_prev = loss
            if(i % 100 == 0 and plot_on == True):
                rec_loss.append(loss)
                if(i == 10000):
                    plt.plot(rec_loss, 'ro')
                    plt.show()
    np.save('model.npy', W)
    real_W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), Y)
    np.save('model.npy', real_W)
    print("Loss of Ans:", end = '')
    MSE(Y, np.dot(X,real_W))

def gd2(X, Y):
    print('gd2')
    print(X.shape, Y.shape)
    c_real = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
    print('loss of ans: ', end = '')
    MSE(Y, np.dot(X, c_real))

    iteration = 10000
    l = 1
    lr_w = np.zeros((X.shape[1],1))
    c = np.zeros((X.shape[1],1))
    for j in range(iteration):
        g = - np.dot(X.T, Y - np.dot(X, c) )
        lr_w += g ** 2
        c -= l * g / np.sqrt(lr_w)
        if j % 20 == 0:
            print('iter{} loss: '.format(j))
            MSE(Y, np.dot(X, c))

def MSE(Y_real, Y_predict):
    loss = np.linalg.norm(Y_real - Y_predict) ** 2 / Y_real.shape[0]
    print(loss)
    return loss

if __name__ == '__main__':
    n = 9
    train_X, train_Y = read_train_data('train.csv', N = n)
    grad_descent(train_X, train_Y, plot_on = True)
    # gd2(train_X, train_Y)
    test_X = read_test_data('test.csv', N = n)
    W = np.load('model.npy')
    test_Y = np.dot(test_X, W)
    # print(test_Y)
    real_Y = pd.read_csv('ans.csv').as_matrix()
    real_Y = real_Y[:,1:]
    real_Y = real_Y.astype('float')
    print('Loss of test:', end = '')
    MSE(real_Y, test_Y)
    # test_Y = read_ans('ans.csv', N = n)
    # print(test_X, '\n', test_X.shape)
    # print(test_X[0])
