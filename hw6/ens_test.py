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
filename_out = sys.argv[3]
filename_DICT = sys.argv[2]

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
test_X = text_to_index(input_datas)

from keras.preprocessing.sequence import pad_sequences
# padding
padding_length = 200
test_X = pad_sequences(test_X, maxlen = padding_length)

batch_test_data = np.zeros((20000,28829), 'int16')
for i in range(20000):
    for j in test_X[i]:
        batch_test_data[i,j] += 1
test_X2 = batch_test_data.reshape(20000,1,28829)

# DNN Model


model1 = load_model('model1.h5')
model4 = load_model('model4.h5')
model5 = load_model('model5.h5')
model8 = load_model('model8.h5')

testY1 = model1.predict(test_X)[:,199,0].reshape(20000,1)
testY2 = model4.predict(test_X2)[:,:,0].reshape(20000,1)
testY3 = model5.predict(test_X2)[:,:,0].reshape(20000,1)
testY4 = model8.predict(test_X)[:,199,0].reshape(20000,1)

test_concat = np.concatenate([testY1, testY2, testY3, testY4], axis = 1)
print(test_concat.shape)

##### build ens classifier
model = load_model('model.h5')
prediction = model.predict_classes(test_concat)
print(prediction.shape)

with open(filename_out, 'w') as f:

    print('id,label', file = f)
    for i in range(prediction.shape[0]):
        print('%d,%d' % (i,prediction[i]), file = f)
