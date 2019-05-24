from sklearn.cluster import KMeans
from keras.models import load_model
import numpy as np
import csv
from sklearn.decomposition import PCA
import sys

# bash cluster.sh <images path> <test_case.csv path> <prediction file path>

filename = sys.argv[2]
filename_out = sys.argv[3]

X = np.load('img.npy')
X = X.astype('float32') / 255
X = np.reshape(X, (40000,32,32,3))
print("SHAPE X:", X.shape)

# load and predict
encoder0 = load_model('encoder.h5')
encoder0.summary()
encode_img0 = encoder0.predict(X)
encode_img = encode_img0.reshape(encode_img0.shape[0], -1)

# PCA
pca = PCA(n_components=300, copy=False, whiten=True, svd_solver='full', random_state=100).fit_transform(encode_img)
print("SHAPE PCA:", pca.shape)

k = KMeans(n_clusters = 2, random_state = 100).fit(pca)
print('ksum:',k.labels_.sum())

r = csv.reader(open(filename))
l = list(r)[1:]
ans = []
for i in l:
    ans.append(int(k.labels_[int(i[1])-1] == k.labels_[int(i[2])-1]))

with open(filename_out, 'w') as f:
    print('id,label', file = f)
    for i in range(len(ans)):
        print('%d,%d' % (i,ans[i]), file = f)