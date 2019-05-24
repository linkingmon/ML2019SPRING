from skimage import io
import numpy as np
import os
import sys

# bash pca.sh <images path> <input image> <reconstruct image>
# e.g. bash  pca.sh  Aberdeen/   87.jpg   87_reconstruct.jpg

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
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    path3 = sys.argv[3]
    imgs_path = os.path.join(path1, '*')
    m = np.load('mean_face.npy')
    U = np.load('U.npy')
    S = np.load('S.npy')
    V = np.load('V.npy')
    img = os.path.join(path1, path2)
    x = read_img(img, m)
    reconstruct(x, U, m, 5, path3)
