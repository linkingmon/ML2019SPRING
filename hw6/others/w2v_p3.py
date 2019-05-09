import jieba
import numpy as np
import pandas as pd
from gensim.models import word2vec
import re

filename_X = 'train_x.csv'
filename_Y = 'train_y.csv'
filename_DICT = 'dict.txt.big'

punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")
jieba.load_userdict(filename_DICT)
input_datas = [list(jieba.cut(t)) for t in pd.read_csv(filename_X).values[:,1]]
for id, word_list in enumerate(input_datas):
    word_list2 = []
    for word in word_list:
        if type(punctuation_search.match(word,0)) == type(None):
            word_list2.append(word)
    input_datas[id] = word_list2

model = word2vec.Word2Vec(input_datas, size = 250, window = 5, min_count = 5, workers = 4, iter = 10, sg = 1)
model.save("dcard_word2vec_prob3.model")