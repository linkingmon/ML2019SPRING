import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
import sys
import pandas as pd
from keras.callbacks import History ,ModelCheckpoint
import re

punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")
filename_X = sys.argv[1]
filename_Y = sys.argv[2]
filename_DICT = sys.argv[4]

jieba.load_userdict(filename_DICT)
w2v_model = word2vec.Word2Vec.load("dcard_word2vec_prob3.model")

# embedding layer
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1
    
embedding_layer = Embedding(input_dim = embedding_matrix.shape[0],
                            output_dim = embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            trainable = False
                            )


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
for id, word_list in enumerate(input_datas):
    word_list2 = []
    for word in word_list:
        if type(punctuation_search.match(word,0)) == type(None):
            word_list2.append(word)
    input_datas[id] = word_list2
label = to_categorical(np.array([int(i) for i in pd.read_csv(filename_Y).values[:,1]])).reshape(120000,1,2)
train_data = text_to_index(input_datas)


# padding
padding_length = 200
train_data = pad_sequences(train_data, maxlen = padding_length)


#RNN model
model = Sequential()
model.add(embedding_layer)  
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(TimeDistributed(Dense(256, activation='relu')))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))
model.summary()


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',  metrics=['accuracy'])

check_save = ModelCheckpoint('model9/model_{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}.h5')

hist = model.fit(x = train_data, y = label, batch_size = 128, epochs = 4, validation_split = 0.12, callbacks = [check_save])

import matplotlib.pyplot as plt
##### figure - loss
plt.gcf().clear()
plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'], 'b', label = 'train loss')
plt.plot(hist.history['val_loss'], 'r', label = 'val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.savefig('model9/loss')

##### figure - acc
plt.gcf().clear()
plt.figure(figsize=(12,8))
plt.plot(hist.history['acc'], 'b', label = 'train acc')
plt.plot(hist.history['val_acc'], 'r', label = 'val acc')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('acc')
plt.savefig('model9/acc')