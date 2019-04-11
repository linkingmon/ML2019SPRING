import numpy as np
import pandas as pd
import sys
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Concatenate, Dense

##### filename of test.csv
##### filename of output-file
inp = sys.argv[1]
out = sys.argv[2]

##### read test data
data = pd.read_csv(inp)
feature = np.array([row.split(" ") for row in data["feature"].tolist()],dtype=np.float32)
feature = np.reshape(feature, (np.shape(feature)[0] , 48,48,1))

def normalize(x):
    x = x/255.
    return x
    
xTest = normalize(feature)


##### load ens model
m1 = load_model('model1.h5')
m2 = load_model('model2.h5')
m3 = load_model('model3.h5')
m4 = load_model('model4.h5')

##### process test data
yTest11 = m1.predict(xTest)
yTest12 = m2.predict(xTest)
yTest13 = m3.predict(xTest)
yTest14 = m4.predict(xTest)

concat = np.concatenate([yTest11, yTest12, yTest13, yTest14], axis = 1)

##### load ens classifier
model_all = load_model('model000.h5')
yTest = model_all.predict(concat)
output = np.argmax(yTest, axis = 1)
index = [i for i in range(0,output.size)]
outCsv = pd.DataFrame({'id' : index, 'label' : output})
outCsv.to_csv(out, index=0)