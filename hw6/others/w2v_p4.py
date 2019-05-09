import numpy as np
import pandas as pd
from gensim.models import word2vec

filename_X = 'test_x.csv'
filename_DICT = 'dict.txt.big'
filename_out = 'ansens2.csv'

input_datas = [list(i) for i in pd.read_csv(filename_X).values[:,1]]

model = word2vec.Word2Vec(input_datas, size = 250, window = 5, min_count = 5, workers = 4, iter = 10, sg = 1)
model.save("dcard_word2vec_prob4.model")