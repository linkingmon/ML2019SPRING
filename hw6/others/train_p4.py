import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Embedding, Dense, BatchNormalization, LeakyReLU, Dropout
from keras.callbacks import History, ModelCheckpoint
import pandas as pd
import sys

filename_X = sys.argv[1]
filename_Y = sys.argv[2]
filename_DICT = sys.argv[4]

jieba.load_userdict(filename_DICT)
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
print(embedding_matrix.shape)
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

input_datas = [list(i) for i in pd.read_csv(filename_X).values[:,1]]
label = to_categorical(np.array([int(i) for i in pd.read_csv(filename_Y).values[:,1]])).reshape(120000,1,2)
train_data = text_to_index(input_datas)
print(train_data.shape, label.shape)

batch_train_data = np.zeros((120000,embedding_matrix.shape[0]), 'int16')
for i in range(120000):
    for j in train_data[i]:
        batch_train_data[i,j] += 1
batch_train_data = batch_train_data.reshape(120000,1,embedding_matrix.shape[0])
batch_label = label

# DNN Model
model = Sequential()

model.add(Dense(256, input_shape=(1,embedding_matrix.shape[0]), activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.8))

model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.7))

model.add(Dense(16, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.8))

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

check_save = ModelCheckpoint('model5/model_{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}.h5', monitor = 'val_acc', save_best_only = True)
hist = model.fit(x = batch_train_data, y = batch_label, batch_size = 32, epochs = 10, validation_split = 0.1, callbacks = [check_save])

import matplotlib.pyplot as plt
##### figure - loss
plt.gcf().clear()
plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'], 'b', label = 'train loss')
plt.plot(hist.history['val_loss'], 'r', label = 'val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig('model5/loss')

##### figure - acc
plt.gcf().clear()
plt.figure(figsize=(12,8))
plt.plot(hist.history['acc'], 'b', label = 'train acc')
plt.plot(hist.history['val_acc'], 'r', label = 'val acc')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('acc')
plt.savefig('model5/acc')