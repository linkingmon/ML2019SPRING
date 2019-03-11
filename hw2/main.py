##### ignore warnings
import warnings
warnings.filterwarnings('ignore')
##### import
import pandas as pd
import numpy as np
import math

def sigmoid(x):                                 
    return 1 / (1 + np.exp(-x))

class binary_classification:
    def __init__(self, cross_valid):
        self.cross_valid = cross_valid

    def read_train_data(self, filename1, filename2):
        print('read train')
        self.train_X = pd.read_csv(filename1).as_matrix()
        self.train_Y = pd.read_csv(filename2).as_matrix()
        self.train_X = data_processing(self.train_X, Nor = True)
        ##### check shape
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

        self.W = np.dot((mean_C0 - mean_C1), Var_inv)
        self.B = -np.dot(np.dot(mean_C0,Var_inv), mean_C0.T) / 2 + np.dot(np.dot(mean_C1,Var_inv), mean_C1.T) / 2 + np.log(cnt_C0 / cnt_C1)
        
        if self.cross_valid:
            predict_Y = sigmoid( (np.dot(self.valid_X,self.W.T) + self.B) ) >= 0.5
            return success_rate(self.valid_Y, predict_Y)
        else:
            return sigmoid( (np.dot(self.test_X,self.W.T) + self.B) ) >= 0.5

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
            return np.dot(self.test_X, W) > 0

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
    useless = []
    X = np.delete(X, useless, axis=1)
    age = [np.power(X[:,0], s).reshape(-1, 1) for s in [2, 3]]
    fnlwgt = [np.power(X[:,1], s).reshape(-1, 1) for s in [2, 3]]
    cap = [np.power(X[:,3], s).reshape(-1, 1) for s in [2, 3, 4, 5]]
    hours = [np.power(X[:,5], s).reshape(-1, 1) for s in [2, 3]]
    # male = [np.power(X[:,76], s).reshape(-1, 1) for s in [2, 3]]
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
    generative = False
    ll = []

    if cross_valid:
        for i in range(10):
            model = binary_classification(True)
            model.read_train_data('X_train', 'Y_train')
            model.validate(i)
            if generative:
                ll.append(model.generative())
            else:
                model.generative()
                ll.append(model.discrimitive())
        print(ll)
        print('Average valid loss:',np.mean(ll))
    
    else:
        model = binary_classification(False)
        model.read_train_data('X_train', 'Y_train')
        model.read_test_data('X_test')
        if generative:
            test_Y = model.generative()
        else:
            test_Y = model.discrimitive()
        f = open('Y_{}.csv'.format('gen' if generative else 'dis'), 'w')
        f.write('id,label\n')
        for i in range(test_Y.shape[0]):
            f.write(repr(i + 1) + "," + repr(int(test_Y[i])) + "\n")
        f.close()