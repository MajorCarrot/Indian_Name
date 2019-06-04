#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:00:09 2019

@author: adithya
"""

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import Adam

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def model1(vocablen, buff_length):
    model = Sequential()
    model.add(SimpleRNN(256,input_shape=(buff_length, vocablen)))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(units=vocablen, activation='softmax'))
    model.summary()
    return model

def one_hot_buffer(X, char2idx, vocablen, buff_length):
    Tx = len(X)
    Xoh = np.zeros((Tx, buff_length, vocablen))
    for idx in range(Tx):
        name = str(X[idx])
        for i in range(buff_length):
            ch = name[i]
            Xoh[idx, i, char2idx[ch]] = 1
    
    return Xoh

def one_hot(Y, char2idx, vocablen, buff_length):
    Ty = len(Y)
    Yoh = np.zeros((Ty, vocablen))
    for idx in range(Ty):
        Yoh[idx, char2idx[Y[idx]]] = 1
    return Yoh

def trainer(X, vocab, char2idx, buff_length, no_epochs=200):
    model = model1(len(vocab), buff_length)
    opt = Adam(lr = .0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    Tx = len(X)
    indices = range(Tx)
    X_train = []
    Y_train = []
    for index in indices:
        name = str(X[index])
        for chIndex in range(len(name)): 
            if chIndex >= buff_length:
                X_train.append(name[chIndex - buff_length : chIndex ])
            else:
                toAppend=''
                for i in range(buff_length):
                    if i < buff_length - chIndex:
                        toAppend = toAppend + '0' 
                    else:
                        toAppend = toAppend + name[i - buff_length + chIndex]
                X_train.append(toAppend)
            Y_train.append(name[chIndex])
                   
    for i in range(20):
        print ((X_train[i] + ' : '+ Y_train[i]) )

    X_train_oh = np.copy(one_hot_buffer(X_train, char2idx, len(vocab), buff_length))
    Y_train_oh = np.copy(one_hot(Y_train, char2idx, len(vocab), buff_length))
    
    X_train_oh, Y_train_oh = unison_shuffled_copies(X_train_oh, Y_train_oh)
    # print(X_train_oh.shape,':',Y_train_oh.shape)

    model.fit(x=X_train_oh, y=Y_train_oh, epochs=no_epochs)
    model.save('../model/'+input("Enter the name for .h5 file") + '.h5')

