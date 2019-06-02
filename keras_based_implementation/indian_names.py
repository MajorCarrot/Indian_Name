#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:00:09 2019

@author: adithya
"""
import string
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

buff_length = 3
maxlen = 20


def model1(vocab_len):
    model = Sequential()
    model.add(LSTM(128,input_shape=(buff_length, vocab_len)))
    model.add(Dense(units=60, activation='relu'))
    model.add(Dense(units=vocab_len, activation='softmax'))
    model.summary()
    return model

def one_hot_buffer(X, char2idx, vocablen):
    Tx = len(X)
    Xoh = np.zeros((Tx, buff_length, vocablen))
    for idx in range(Tx):
        name = str(X[idx])
        for i in range(buff_length):
            ch = name[i]
            Xoh[idx, i, char2idx[ch]] = 1
    
    return Xoh

def one_hot(Y, char2idx, vocablen):
    Ty = len(Y)
    Yoh = np.zeros((Ty, vocablen))
    for idx in range(Ty):
        Yoh[idx, char2idx[Y[idx]]] = 1
    return Yoh

def trainer(X, vocab, char2idx, no_epochs=1, batch_size=10):
    model = model1(len(vocab))
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    for epn in range(no_epochs):
        np.random.seed(1 + epn)
        Tx = len(X)
        indices = np.random.randint(0, Tx, batch_size)
        X_train = []
        Y_train = []
        for index in indices:
            name = str(X[index])
            for chIndex in range(len(name) - 1): 
                if chIndex >= buff_length - 1:
                    X_train.append(name[chIndex - buff_length + 1: chIndex + 1])
                    Y_train.append(name[chIndex + 1])
        
        for i in range(len(X_train)):
            print ((X_train[i] + ' : '+ Y_train[i]) )

        X_train_oh = np.copy(one_hot_buffer(X_train, char2idx, len(vocab)))
        Y_train_oh = np.copy(one_hot(Y_train, char2idx, len(vocab)))
        
        print(X_train_oh.shape,':',Y_train_oh.shape)
        model.fit(x=X_train_oh, y=Y_train_oh)
    
    model.save('name_model.h5')

names = []

with open("Name.txt") as f:
    names = f.readlines()
    f.close()
names = [x.lower().strip() + '\n' for x in names]
    
vocab=dict(zip(string.ascii_lowercase, range(0,26)))
vocab['\n']=26

char2idx = vocab
idx2char = np.array(list(vocab))

trainer(names, list(vocab), char2idx)