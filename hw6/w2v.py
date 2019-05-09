import jieba
import numpy as np
import pandas as pd
from gensim.models import word2vec

filename_X = sys.argv[1]
filename_Y = sys.argv[2]
filename_DICT = sys.argv[4]

jieba.load_userdict(filename_DICT)
input_datas = [list(jieba.cut(t)) for t in pd.read_csv(filename_X).values[:,1]]

model = word2vec.Word2Vec(input_datas, size = 250, window = 5, min_count = 5, workers = 4, iter = 10, sg = 1)
model.save("dcard_word2vec.model")