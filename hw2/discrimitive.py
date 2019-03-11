##### import
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):                                 
    return 1 / (1 + np.exp(-x))

class binary_classification:
    def __init__(self, cross_valid):
        self.cross_valid = cross_valid

    def read_train_data(self, filename1, filename2):
        print('read train')
        self.train_X = pd.read_csv(filename1).as_matrix()
        self.train_Y = pd.read_csv(filename2).as_matrix()
        # plt.plot(self.train_X[:,5:6], self.train_Y, 'ro')
        # plt.savefig('obs/hours')
        self.train_X = data_processing(self.train_X)
        ##### check shape
        print(self.train_X.shape,self.train_Y.shape)

    def validate(self,num):
        len3 = self.train_X.shape[0] // 10
        idx = [t for t in range(num*len3,len3 + num*len3)]
        self.valid_X = self.train_X[idx, :]
        self.train_X = np.delete(self.train_X, idx, axis = 0)
        self.valid_Y = self.train_Y[idx, :]
        self.train_Y = np.delete(self.train_Y, idx, axis = 0)

    def discrimitive(self):
        W = np.zeros((self.train_X.shape[1], 1))
        ##### data processing
        ##### ada_grad
        lr_w = np.ones((self.train_X.shape[1], 1)) * 1e-4
        l_rate = 1
        lamda = 0.01
        iteration = 4000
        correct = correct2 = 0
        for i in range(iteration):
            Z = np.dot(self.train_X, W) > 0
            correct = success_rate(self.train_Y, Z)
            if self.cross_valid:
                Z2 = np.dot(self.valid_X, W) > 0
                correct2 = success_rate(self.valid_Y, Z2)
                if i % 10 == 0:
                    print('Iter{} Correct rate: {}%  {}%'.format(i, correct*100, correct2*100))
            else:
                if i % 10 == 0:
                    print('Iter{} Correct rate: {}%'.format(i, correct*100))
            #### gradient
            w_g = - np.dot(self.train_X.T, self.train_Y - sigmoid(np.dot(self.train_X, W)) ) + 2 * lamda * W
            ##### adagrad
            lr_w = lr_w + w_g**2
            W = W - l_rate * w_g / np.sqrt(lr_w)

        if self.cross_valid:
            return correct2
        else:
            np.save("./model_discrimitive.npy", W)

def data_processing(X):
    ##### normalize
    mean = np.mean(X, axis = 0).astype(np.float32)
    dev = np.std(X, axis = 0)
    np.save("./mean.npy", mean)
    np.save("./dev.npy", dev)
    X = (X - mean) / dev
    ##### data processing
    useless = []
    X = np.delete(X, useless, axis=1)
    age = [np.power(X[:,0], s).reshape(-1, 1) for s in [2, 3]]
    fnlwgt = [np.power(X[:,1], s).reshape(-1, 1) for s in [2, 3]]
    cap = [np.power(X[:,3], s).reshape(-1, 1) for s in [2, 3, 4, 5]]
    hours = [np.power(X[:,5], s).reshape(-1, 1) for s in [2, 3]]
    X = np.concatenate([X] + age + fnlwgt + cap + hours, axis = 1) 
    ##### adding bias
    row = X.shape[0]
    bias = np.ones(shape = (row, 1), dtype=np.float32)
    return np.concatenate((bias, X), axis=1)
    
def success_rate(real_Y, predict_Y):
    success_rate = np.sum( (real_Y - predict_Y) == 0) / real_Y.shape[0]
    return success_rate

if __name__ == '__main__':
    
    cross_valid = False
    ll = []

    if cross_valid:
        for i in range(10):
            model = binary_classification(True)
            model.read_train_data('X_train', 'Y_train')
            model.validate(i)
            ll.append(model.discrimitive())
        print(ll)
        print('Average valid loss:',np.mean(ll))
    
    else:
        model = binary_classification(False)
        model.read_train_data('X_train', 'Y_train')
        model.discrimitive()