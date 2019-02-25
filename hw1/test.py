##### import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def read_test_data(filename, N):
    ##### big5 encoding
    text = open(filename, 'r', encoding='big5')
    raw_data = pd.read_csv(text, header = None).as_matrix()
    ##### first 2 columns are not data
    data = raw_data[:, 2:]
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
    test_X = np.array(X)
    print('Shape of test X:', test_X.shape)
    return test_X

def normal(test_X):
    ##### normalize both train ans test data (with same mu ans sig)
    mu = np.loadtxt('normal_mu.txt')
    sig = np.loadtxt('normal_sig.txt')
    print(mu)
    for i in range(test_X.shape[1]-1):
        test_X[:, i] = (test_X[:, i] - mu[i]) / sig[i]
        test_X[:, i] = (test_X[:, i] - mu[i]) / sig[i]
    return test_X

if __name__ == '__main__':
    ##### file name
    infile = sys.argv[1]
    outfile = sys.argv[2]
    ##### read data
    test_X = read_test_data(infile, N = 9)
    ##### load model
    c = np.load('model21.npy')
    ##### calculate test_Y
    test_Y = np.dot(test_X, c)
    ##### write test_Y
    with open(outfile, 'w') as f:
        f.write('id,value\n')
        for i in range(test_Y.shape[0]):
            f.write('id_{},{}\n'.format(i,test_Y[i][0]))
        f.close()