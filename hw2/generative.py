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

    def generative(self):
        sum_C0 = np.zeros((1, self.train_X.shape[1]))
        sum_C1 = np.zeros((1, self.train_X.shape[1]))
        sumV_C0 = np.zeros((self.train_X.shape[1], self.train_X.shape[1]))
        sumV_C1 = np.zeros((self.train_X.shape[1], self.train_X.shape[1]))
        cnt_C0 = 0
        cnt_C1 = 0
        for i in range(self.train_X.shape[0]):
            if self.train_Y[i] == 1:
                sum_C0 += self.train_X[i:i+1]
                sumV_C0 += np.dot(self.train_X[i:i+1].T, self.train_X[i:i+1])
                cnt_C0 += 1
            else:
                sum_C1 += self.train_X[i:i+1]
                sumV_C1 += np.dot(self.train_X[i:i+1].T, self.train_X[i:i+1])
                cnt_C1 += 1

        mean_C0 = sum_C0 / cnt_C0
        mean_C1 = sum_C1 / cnt_C1

        meanV_C0 = sumV_C0 / cnt_C0
        meanV_C1 = sumV_C1 / cnt_C1

        Var_C0 = meanV_C0 - np.dot(mean_C0.T, mean_C0)
        Var_C1 = meanV_C1 - np.dot(mean_C1.T, mean_C1)

        Var = (Var_C0*cnt_C0 + Var_C1*cnt_C1) / (cnt_C0 + cnt_C1)
        Var_inv = np.linalg.pinv(Var)

        W = np.dot((mean_C0 - mean_C1), Var_inv)
        B = -np.dot(np.dot(mean_C0,Var_inv), mean_C0.T) / 2 + np.dot(np.dot(mean_C1,Var_inv), mean_C1.T) / 2 + np.log(cnt_C0 / cnt_C1)
        
        if self.cross_valid:
            predict_Y = sigmoid( (np.dot(self.valid_X,W.T) + B) ) >= 0.5
            return success_rate(self.valid_Y, predict_Y)
        else:
            np.save("./model_generative_W.npy", W)
            np.save("./model_generative_B.npy", B)

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
            ll.append(model.generative())
        print(ll)
        print('Average valid loss:',np.mean(ll))
    
    else:
        model = binary_classification(False)
        model.read_train_data('X_train', 'Y_train')
        model.generative()