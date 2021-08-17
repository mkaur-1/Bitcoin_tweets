import numpy as np 
import pandas as pd 
from textblob import TextBlob
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        file = '/kaggle/input/bitcoin-tweets-20160101-to-20190329/tweets.csv'
df = pd.read_csv(file, sep=';',nrows=1300)
print(df.head())
data = df.drop(['replies','likes','retweets','timestamp','url','id','user','fullname'],axis = 1)
data.reset_index(drop=True, inplace=True)
data.head()
tweets = data['text']
print(tweets[2])
pip install whatthelang
from whatthelang import WhatTheLang

wtl = WhatTheLang()
L=[]
for row in data['text']:
    if len(row)!=0:
        L.append(wtl.predict_lang(row))
    else:
        L.append(None)
        
data['lang'] = L
data.head()
data = data[data["lang"] == 'en']
data.head()
import nltk
import re
from nltk.corpus import stopwords

def text_cleaning(text):
    forbidden_words = set(stopwords.words('english'))
    text = ' '.join(text.split('.'))
    text = re.sub('\/',' ',text)
    text = text.strip('\'"')
    text = re.sub(r'@([^\s]+)',r'\1',text)
    text = re.sub(r'\\',' ',text)
    text = text.lower()
    text = re.sub('[\s]+', ' ', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'((http)\S+)','',text)
    text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
    text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
    text = [word for word in text.split() if word not in forbidden_words]
    return ' '.join(text)

data['text'] = data['text'].apply(lambda text: text_cleaning(text))
data.sample(3)
from textblob import TextBlob

def sentiment(txt):
    return TextBlob(txt).sentiment.polarity

data['sentiment'] = data['text'].apply(lambda txt: sentiment(txt))      # new column of sentiment

data.sample(10)
data.to_csv('my_clean_tweets.csv', sep = ';',index = False)

tweets=pd.read_csv('my_clean_tweets.csv', sep=';')
tweets.sample(10)
from numpy.random import RandomState

rng = RandomState()
train_data = tweets.sample(frac=0.8, random_state=rng)
test_data = tweets.loc[~tweets.index.isin(train_data.index)]
print('La taille des données d entrinement:',len(train_data))
print('La taille des données de test:',len(test_data))
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Activation,Embedding,Bidirectional
max_features = 20000  # Only consider the top 20k words
maxlen = 200
train_data['flag'] = 'TRAIN'
test_data['flag'] = 'TEST'


total_docs = pd.concat([train_data,test_data],axis = 0,ignore_index = True)
phrases = total_docs['text'].tolist()

total_docs.sample(10)
from keras.preprocessing.text import one_hot
vocab_size = 50000
encoded_phrases = [one_hot(d, vocab_size) for d in phrases]
total_docs['Phrase'] = encoded_phrases
train_data = total_docs[total_docs['flag'] == 'TRAIN']
test_data = total_docs[total_docs['flag'] == 'TEST']
x_train = train_data['Phrase']
y_train = train_data['sentiment']
x_val = test_data['Phrase']
y_val = test_data['sentiment']
print(total_docs['Phrase'][23])
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
model = Sequential()
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
model.add(inputs)
model.add(Embedding(50000, 32))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))

# Add a classifier
model.add(Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=["accuracy"])

model.fit(x_train, y_train, 
          batch_size=128, 
          epochs=20, 
          validation_data=(x_val, y_val),
          validation_steps=20)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

sample_text = ('Bitcoin just lost half its value overnight. Sorry all you savvy investors ')
vocab_size = 50000

model.predict(one_hot(sample_text, vocab_size))
