import math
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.utils import to_categorical
from keras.models import load_model

num_classes = 7
X_shape = (-1,48,48,1)

labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral',
}


def remove_till_layer(model, layer_name):
    while model.layers[len(model.layers)-1].name != layer_name:
        model.pop()
    return model

def get_random_img(X, y, id):
    while True:
        i = np.random.randint(0, len(X))
        label_id = y[i].argmax()
        if label_id == id:
            img = X[i].reshape(X_shape)
            break
    return img, label_id

def get_random_correct_img(X, y, model, id):
    found = False
    while found == False:
        img, label_id = get_random_img(X, y, id)
        pred = model.predict(img)[0]
        if pred.argmax() == label_id:
            found = True
    return img, label_id

def generate_conv_layer_models():
    conv_models = []

    conv_models.append(remove_till_layer(load_model('model1.h5'), 'conv2d_1'))
    return conv_models

def plot_hidden_layers(model, img, title):
    to_visual = model.predict(img)
    to_visual = to_visual.reshape(to_visual.shape[1:])

    _ = plt.figure()
    _ = plt.suptitle(title)
    sub_plot_height = math.ceil(np.sqrt(to_visual.shape[2]))
    
    for i in range(to_visual.shape[2]):
        ax = plt.subplot(sub_plot_height, sub_plot_height, i+1)
        _ = plt.axis('off')
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])
        _ = ax.set_aspect('equal')
        _ = plt.imshow(to_visual[:, :, i], cmap='inferno')
        _ = plt.savefig('fig2_2.jpg')

def gen_models_and_visualize(X, y):
    full_model = load_model('model1.h5')
    conv_models = generate_conv_layer_models()

    img, label_id = get_random_correct_img(X, y, full_model, 3)
    # plt.imshow(img.reshape(48,48))
    # plt.savefig('filters_original_{}'.format(labels[3]))
    # plt.title(labels[label_id])

    for model in conv_models:
        plot_hidden_layers(model, img, 'conv2d_1 given image 7 (Happy)')

def read_train(filename):
    print('Start reading train data')
    data = pd.read_csv(filename).as_matrix()
    train_Y = to_categorical(data[:, 0], 7)
    train_X = np.array([np.array(line.split(' ')).astype('float').reshape(48, 48, 1) for line in data[:, 1] ]) / 255
    return train_X, train_Y

if __name__ == '__main__':
    X, y = read_train('fig.csv')
    gen_models_and_visualize(X, y)