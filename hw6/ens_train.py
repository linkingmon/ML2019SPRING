import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import History, ModelCheckpoint
import pandas as pd
from keras.models import load_model
import sys

filename_X = sys.argv[1]
filename_Y = sys.argv[2]
filename_out = sys.argv[3]
filename_DICT = sys.argv[4]

jieba.load_userdict(filename_DICT)
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

input_datas = [list(jieba.cut(i)) for i in pd.read_csv(filename_X).values[:,1]]
train_Y = to_categorical(np.array([int(i) for i in pd.read_csv(filename_Y).values[:,1]])).reshape(120000,2)
train_X = text_to_index(input_datas)

from keras.preprocessing.sequence import pad_sequences
# padding
padding_length = 200
train_X = pad_sequences(train_X, maxlen = padding_length)

batch_train_data = np.zeros((120000,28829), 'int16')
for i in range(120000):
    for j in train_X[i]:
        batch_train_data[i,j] += 1
train_X2 = batch_train_data.reshape(120000,1,28829)
print(train_X.shape, train_X2.shape, train_Y.shape)

# DNN Model


model1 = load_model('model1.h5')
model4 = load_model('model4.h5')
model5 = load_model('model5.h5')
model8 = load_model('model8.h5')

# testY1 = model1.predict_classes(train_X)[:,0].reshape(120000,1)
# testY2 = model4.predict(train_X2)[:,:,0].reshape(120000,1)
# testY3 = model5.predict(train_X2)[:,:,0].reshape(120000,1)
# testY4 = model8.predict_classes(train_X)[:,0].reshape(120000,1)
testY1 = model1.predict(train_X)[:,199,0].reshape(120000,1)
testY2 = model4.predict(train_X2)[:,:,0].reshape(120000,1)
testY3 = model5.predict(train_X2)[:,:,0].reshape(120000,1)
testY4 = model8.predict(train_X)[:,199,0].reshape(120000,1)
print(testY1.shape, testY2.shape, testY3.shape, testY4.shape)
train_concat = np.concatenate([testY1, testY2, testY3, testY4], axis = 1)
print(train_concat.shape)

##### build ens classifier
model = Sequential()
model.add( Dense(units = 2, input_shape = (4,), activation = 'softmax') )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

##### callbacks
check_save = ModelCheckpoint('model/modele_{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}.h5', monitor = 'val_acc', save_best_only = True)

##### fit
model.fit( train_concat, train_Y, batch_size = 128, validation_split = 0.5, epochs = 40, callbacks = [check_save])
