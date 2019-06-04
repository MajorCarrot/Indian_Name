#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:00:09 2019

@author: adithya
"""
import string
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, SimpleRNN
from keras.optimizers import Adam
from keras.regularizers import l2
import name_train

buff_length = 5
maxlen = 20

def predicter(s, char2idx, idx2char, vocablen, model_path):
    model = model = load_model(model_path)
    if len(s) < buff_length:
        while (len(s) - buff_length) != 0:
            s = '0' + s
    
    while s[len(s) - 1] != '\n' and len(s) < maxlen:
        name_list = []
        temp = ''

        temp = s[len(s) - buff_length:]
            
        name_list.append(temp)
        s = s + idx2char[int(np.argmax(model.predict(name_train.one_hot_buffer(name_list, char2idx, vocablen, buff_length))))]
        
    print (s.strip('0'))
    

names = []

with open("../data/Name.txt") as f:
    names = f.readlines()
    f.close()
names = [x.lower().strip() + '\n' for x in names]
    
vocab=dict(zip(string.ascii_lowercase, range(0,26)))
vocab['0']=26
vocab['\n']=27

char2idx = vocab
idx2char = np.array(list(vocab))

# name_train.trainer(names, list(vocab), char2idx, buff_length)
predicter('ra', char2idx, idx2char, len(vocab), '../model/name_model_20000.h5')

