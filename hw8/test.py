from mobilenet3 import MobileNet
import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
import sys

inp = sys.argv[1]
out = sys.argv[2]

# read test data
data = pd.read_csv(inp)
feature = np.array([row.split(" ")
                    for row in data["feature"].tolist()], dtype=np.float32)
feature = np.reshape(feature, (np.shape(feature)[0], 48, 48, 1))
print(feature.shape)


def normalize(x):
    x = x/255.
    return x


xTest = normalize(feature)
m = MobileNet()
L_n = []
with open('weights', 'rb') as f:
    L_n = pickle.load(f)
m.set_weights(L_n)
m.summary()

y = m.predict(xTest)

# load ens classifier
output = np.argmax(y, axis=1)
index = [i for i in range(0, output.size)]
outCsv = pd.DataFrame({'id': index, 'label': output})
outCsv.to_csv(out, index=0)
