# -*- coding: utf-8 -*-

from __future__ import print_function
import csv
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
import operator
import re

max_vocabulary = 20000.
maxlen = 140  
batch_size = 256

def preprocess_words(words):
    # Remove consecutive period symbols
    words = re.sub(r"[\. ][\. ]+", " . ", words)
    # Replace word+comma with word [space] comma
    words = re.sub(r",", " , ", words)
    # Replace word+parenthesis with word [space] parenthesis.
    words = re.sub(r"[\(\)]", " \1 ", words)
    return words.split()

def load_twitter(filename):
    labels = []
    tweets = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Column names = ItemID,Sentiment,SentimentSource,SentimentText
            labels.append(int(row['Sentiment']))
            words = row['SentimentText'].strip().lower()
            tweets.append(preprocess_words(words))
    return np.array(tweets), np.array(labels)


tweets, labels = load_twitter('Sentiment_Analysis_Dataset.csv')

# randomize tweets and create training/test sets
np.random.seed(231)
rand_idx = np.random.permutation(len(labels))
rand_idx[:-10000]
# select last 10000 tweets as test set
tweets_training = tweets[rand_idx[:-10000]]
labels_training = labels[rand_idx[:-10000]]
tweets_test = tweets[rand_idx[-10000:]]
labels_test = labels[rand_idx[-10000:]]

def build_vocabulary(tweets):
    vocab = dict()
    for t in tweets:
        for word in t:
            if word.startswith('@'): # ignore twitter username
                continue
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    # sort vocabulary by count
    vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    # keep only top max_vocabulary ones
    vocab = vocab[:max_vocabulary]
    vocab.append(('<unk>', 0))
    return vocab

vocabulary = build_vocabulary(tweets_training)

def save_vocab(vocabulary):
    with open('vocab.txt', 'w') as vf:
        for v in vocabulary:
            vf.write(v[0])
            vf.write('\t')
            vf.write(str(v[1]))
            vf.write('\n')

save_vocab(vocabulary)

def create_vocab_index(vocab):
    vocab_idx = dict()
    v_id = 0
    for v in vocab:
        vocab_idx[v[0]] = v_id
        v_id += 1
    return vocab_idx

vocab_word_to_id = create_vocab_index(vocabulary)
vocab_id_to_word = [(idx,word) for (word,idx) in vocab_word_to_id.items()]

def transcode_words(sents, vocab_index):
    coded_words = [[vocab_index[w] if w in vocab_index else vocab_index['<unk>'] for w in words ] for words in sents]
    return coded_words

tweets_training_to_id = transcode_words(tweets_training, vocab_word_to_id)
tweets_test_to_id = transcode_words(tweets_test, vocab_word_to_id)

# transform from word to word_id first!!
print('Pad sequences (samples x time)')
tweets_training_to_id_padded = sequence.pad_sequences(tweets_training_to_id, maxlen=maxlen)
tweets_test_to_id_padded = sequence.pad_sequences(tweets_test_to_id, maxlen=maxlen)
print('features shape:', tweets_training_to_id_padded.shape)
# turn label to one-hot
labels_training_onehot = to_categorical(labels_training, num_classes=2)
labels_test_onehot = to_categorical(labels_test, num_classes=2)

def build_model():
    print('Building model...')
    model = Sequential()
    model.add(Embedding(max_vocabulary, 300))
    model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train(model):
    model.fit(tweets_training_to_id_padded, labels_training_onehot,
              batch_size=batch_size,
              epochs=10,
              validation_data=(tweets_test_to_id_padded, labels_test_onehot))
    score, acc = model.evaluate(tweets_test_to_id_padded, labels_test_onehot, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

tweet_classify_model = build_model()
train(tweet_classify_model)
tweet_classify_model.save('tweet_model.pkl')

import tensorflow as tf
print(tf.__version__)

from keras.models import load_model
pre_trained_model = load_model('tweet_model.pkl')
pre_trained_model.summary()