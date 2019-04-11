import numpy as np
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
from keras.utils import to_categorical

def predict(input):
    y = model.predict(input[:,:,:,0:1])
    return y

def segmentation(input):
    return slic(input)

def get_lime(num):
    explainer = lime_image.LimeImageExplainer()
    image = img_list[num]
    explaination = explainer.explain_instance(image = image, classifier_fn = predict, segmentation_fn = segmentation)
    image, mask = explaination.get_image_and_mask(label=num, positive_only=False, hide_rest=False, num_features=5, min_weight=0.0)
    return image

if __name__ == '__main__':
    ##### read data
    data = pd.read_csv('fig.csv').as_matrix()
    x_train = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ]) / 255
    x_train = x_train.reshape(x_train.shape[0:3])
    x_train_rgb = np.stack((x_train,x_train,x_train), axis = 3)

    ##### laod model
    model = load_model('model1.h5')
    
    label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprirse', 'Neutral']
    img_list = []

    for i in range(7):
        image = x_train_rgb[i]
        img_list.append(image)

    for i in range(7):
        image = get_lime(i)
        plt.title(label_list[i])
        plt.imshow(image)
        plt.savefig('fig3_{}.jpg'.format(i))