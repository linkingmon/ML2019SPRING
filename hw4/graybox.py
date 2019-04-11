import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.utils import to_categorical

label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprirse', 'Neutral']  

def see(num, box_length=5):
    image = x_train[num]
    image_list = []
    for i in range(48 - box_length):
        for j in range(48 - box_length):
            image_cp = image.copy()
            image_cp[i:box_length+i+1,j:box_length+j+1] = 0.61
            image_list.append(image_cp)
    x_boxx = np.stack(image_list,axis = 0)
    y_boxx = model.predict(x_boxx)[:,3]
    prob = model.predict(image.reshape(1,48,48,1))[:,3][0]
    prob_diff = abs(y_boxx-prob)
    prob_diff = prob_diff.reshape(48-box_length,48-box_length)
    G = np.zeros((48,48))
    for i in range(48 - box_length):
        for j in range(48 - box_length):
            G[i:box_length+i+1,j:box_length+j+1] += prob_diff[i,j]
    plt.figure()
    plt.subplot(1,2,1)
    plt.title(label_list[num])
    plt.imshow(image.reshape(48,48))
    plt.subplot(1,2,2)
    plt.title('graybox')
    plt.imshow(G)
    plt.savefig('fig4_{}'.format(num))

if __name__ == '__main__':
    ##### read data
    data = pd.read_csv('fig.csv').as_matrix()
    x_train = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ]) / 255
    
    ##### laod model
    model = load_model('model1.h5')
 
    for i in range(7):
        see(i, box_length=5)