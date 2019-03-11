import pandas as pd
import numpy as np
import sys

def read_test_data(filename1):
    print('read test')
    test_X = pd.read_csv(filename1).as_matrix()
    test_X = data_processing(test_X)
    return test_X
    print(test_X.shape)

def data_processing(X):
    ##### normalize
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
    X = np.concatenate([X] + age + fnlwgt + cap + hours, axis = 1) 
    ##### adding bias
    row = X.shape[0]
    bias = np.ones(shape = (row, 1), dtype=np.float32)
    return np.concatenate((bias, X), axis=1)
    
if __name__ == '__main__':
    test_X = read_test_data(sys.argv[5])

    W = np.load("./model_discrimitive.npy")
    test_Y = np.dot(test_X, W) > 0

    with open(sys.argv[6], 'w') as f:
        f.write('id,label\n')
        for i in range(test_Y.shape[0]):
            f.write(repr(i + 1) + "," + repr(int(test_Y[i])) + "\n")
        f.close()