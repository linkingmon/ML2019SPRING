import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.models import load_model

data = np.load('visualization.npy')
data = data.astype('float32') / 255
data = np.reshape(data, (5000,32,32,3))
print(data.shape)

# pca = PCA(n_components=2, copy=False, whiten=True, svd_solver='full', random_state=100).fit_transform(np.reshape(data,(5000,-1)))
pca = np.load('p2b_pca.npy')
# np.save('p2b_pca.npy', pca)
print(pca.shape)
plt.figure()
plt.plot(pca[:2500,0],pca[:2500,1],'ro',markersize=2)
plt.plot(pca[2500:,0],pca[2500:,1],'bo',markersize=2)
plt.title('PCA')
plt.savefig('p2b_pca.png')

# encoder = load_model('encoder0.h5')
# encoder.summary()
# vae = encoder.predict(data)
vae = np.load('p2b_vae.npy')
# np.save('p2b_vae.npy', vae)
vae = vae[:, :2]
print(vae.shape)
plt.figure()
plt.plot(vae[:2500,0],vae[:2500,1],'ro',markersize=2)
plt.plot(vae[2500:,0],vae[2500:,1],'bo',markersize=2)
plt.title('First 2 dimension of VAE')
plt.savefig('p2b_vae.png')
