import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pandas as pd 

df_train = pd.read_csv('/Users/damodargupta/Downloads/archive/train.txt')
df_test = pd.read_csv('/Users/damodargupta/Downloads/archive/test.txt')
df_val = pd.read_csv('/Users/damodargupta/Downloads/archive/val.txt')

df_final = pd.concat([df_train , df_test , df_val] , axis=0)

emotions = []
for emotion in df_final['emotion']:
    emotions.append(emotion)

sentences = []
for sentence in df_final['sentences']:
  sentences.append(sentence)


# Tokenization and Encoding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_final['sentences'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

model = keras.models.load_model('/Users/damodargupta/Desktop/sentence-emotion analysis model- psyaa/model/sequential_model5.h5')

def prediction(test_sentence):
    # Encoding Test Sentence
    test_sequence = tokenizer.texts_to_sequences([test_sentence])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_len)

    # Emotion Prediction
    predictions = model.predict(padded_test_sequence)
    predicted_emotion = emotions[np.argmax(predictions)]

    print("Test Sentence: ", test_sentence)
    print("Predicted Emotion: ", predicted_emotion)

