import numpy as np
from sklearn.cluster import KMeans
import csv 
from sklearn.decomposition import PCA

x_train = np.load('img.npy')
x_train = np.reshape(x_train, (40000, 32, 32, 3))

x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train,(40000,-1))
print(x_train.shape)

pca = PCA(n_components=400, copy=False, whiten=True, svd_solver='full', random_state=100)
w = pca.fit_transform(x_train)
res = pca.inverse_transform(w)
print(x_train.shape, res.shape)
recons_loss = np.sum( (x_train-res) ** 2 ) / (32*32*3*40000)
print(recons_loss)

k = KMeans(n_clusters = 2, random_state = 100).fit(w)
print('ksum:',k.labels_.sum())
np.save('Label.npy',k.labels_)

filename_out = 'p2_pca.csv'
filename = 'test_case.csv'

r = csv.reader(open(filename))
l = list(r)[1:]
ans = []
for i in l:
    ans.append(int(k.labels_[int(i[1])-1] == k.labels_[int(i[2])-1]))

with open(filename_out, 'w') as f:
    print('id,label', file = f)
    for i in range(len(ans)):
        print('%d,%d' % (i,ans[i]), file = f)