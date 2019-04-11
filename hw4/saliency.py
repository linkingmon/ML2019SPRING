import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras.backend as K

def compile_saliency_function(model):
    inp = model.layers[0].input
    outp = model.layers[-1].output
    max_outp = K.max(outp, axis=1)
    saliency = K.gradients(max_outp, inp)[0]
    saliency = K.abs(saliency)
    max_class = K.argmax(outp, axis=1)
    return K.function([inp], [saliency, max_class])

if __name__ == '__main__':

    ##### load model
    model = load_model('./model1.h5')

    ##### read train data
    data = pd.read_csv('fig.csv').as_matrix()
    y_test = to_categorical(data[:, 0], 7)
    x_test = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ]) / 255

    ##### labels
    tit = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprirse', 'Neutral']
    
    for idx in np.arange(7):    
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(x_test[idx][..., 0])
        ax[0].set_title(tit[idx])
        sal, max_class = compile_saliency_function(model)([x_test[idx].reshape(1,48,48,1), 0])
        image = sal[0].reshape(48,48)
        ax[1].imshow(image, cmap='jet')
        ax[1].set_title('Saliency map')
        plt.savefig('fig1_{}.jpg'.format(idx))

