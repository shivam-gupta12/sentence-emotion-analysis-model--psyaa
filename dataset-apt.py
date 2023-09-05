import csv
import nltk
nltk.download('punkt')
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd


df = pd.read_csv('/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/google_emotions.csv')
dictionary = df.to_dict()
print(dictionary)

with open('/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/google_emotions.csv', 'r') as f:
    reader = csv.reader(f)
    dataset = list(reader)

text_column = [row[0] for row in dataset]
emotion_column = [row[1] for row in dataset]
#print(emotion_column)

    
tokens = [nltk.word_tokenize(sentence) for sentence in text_column]

model = Word2Vec(tokens, window=5, min_count=1, workers=4)

features = np.array([np.mean([model.wv[token] for token in sentence], axis=0) for sentence in tokens])

#print(features)



