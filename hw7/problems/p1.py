from skimage import io
import numpy as np
import os
import sys

# read images & cal. mean & minus mean


def load_data(path):
    imgs = io.imread_collection(path)
    imgs_array = np.array(imgs)
    imgs_array = np.reshape(imgs_array, (415, -1))
    print("SHAPE OF IMGS_ARRAY", imgs_array.shape)
    X = imgs_array.T.astype('float32')
    mean_face = np.mean(X, axis=1)
    np.save('mean_face.npy', mean_face)
    print("SHAPE OF MEAN_Face", mean_face.shape)
    X -= mean_face.reshape(1080000, 1)
    return X, mean_face


def get_SVD(X):
    U, S, V = np.linalg.svd(X, full_matrices=False)
    # U = np.load('U.npy')
    # S = np.load('S.npy')
    # V = np.load('V.npy')
    np.save('U.npy', U)
    np.save('S.npy', S)
    np.save('V.npy', V)
    print(S[:5])
    print(S[:5]/np.sum(S))
    for i in range(5):
        name = 'eigenface_{}.png'.format(i)
        res = -U[:, i]
        M = np.reshape(res, (600, 600, 3))
        M -= np.min(M)
        M /= np.max(M)
        M = (M*255).astype('uint8')
        io.imsave(name, M)
    print("SHAPE OF U S V:", U.shape, S.shape, V.shape)
    return U, S, V


def reconstruct(x, U, mean_face, dim, name):
    mean_face = np.reshape(mean_face, (600, 600, 3))
    u = U[:, :dim]
    w = u.T.dot(x)
    res = u.dot(w)
    M = np.reshape(res, (600, 600, 3))

    M += mean_face
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype('uint8')
    io.imsave(name, M)


def read_img(path, mean_face):
    img = io.imread(path)
    img_array = np.array(img)
    img_array = img_array.reshape((600*600*3,))
    x = img_array.astype('float64')
    x -= mean_face
    return x


if __name__ == '__main__':
    # path1 = sys.argv[1]
    # path2 = sys.argv[2]
    path1 = 'Aberdeen/'
    paths = [10, 30, 50, 70, 90]
    imgs_path = os.path.join(path1, '*')
    X, m = load_data(imgs_path)
    io.imsave('mean_face.png', m.reshape(600, 600, 3))
    U, S, V = get_SVD(X)
    for num in paths:
        path2 = '{}.jpg'.format(num)
        img = os.path.join(path1, path2)
        x = read_img(img, m)
        name = 'resconstruct_img_{}.png'.format(num)
        reconstruct(x, U, m, 5, name)
