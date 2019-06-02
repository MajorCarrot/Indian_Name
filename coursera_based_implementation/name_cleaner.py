#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:22:57 2019

@author: adithya
"""
import pandas as pd
import re

data = pd.read_csv('../data/Indian-Male-Names.csv', header = 0)
data=data[['name']].values.tolist()

clean_data = []

def special_match(strg, search=re.compile(r'[^a-zA-Z]').search):
     return not bool(search(strg))

for datas in data:
#    datas = datas[0]
#    st = datas.split('@')[0].strip()
#    st = st.split('/')[0].strip()
#    st = st.lower().split(',')[0].strip()
#    if ' with ' in st or ' urf ' in st or ' son ' in st:
#        continue
#    if special_match(st):
#        clean_data.append(st[:20] if len(st) > 20 else st)
    
    datas = str(datas[0])
    st = datas.split(' ')[0].strip().lower()
    if special_match(st) and st != '':
        clean_data.append(st[:20] if len(st) > 20 else st)
    

data = pd.read_csv('../data/Indian-Female-Names.csv', header = 0)
data=data[['name']].values.tolist()

for datas in data:
#    datas = str(datas[0])
#    if datas == '':
#        continue
#    st = datas.split('@')[0].strip()
#    st = st.split('/')[0].strip()
#    st = st.lower().split(',')[0].strip()
#    if ' with ' in st or ' urf ' in st or ' son ' in st:
#        continue
#    if special_match(st):
#        clean_data.append(st[:20] if len(st) > 20 else st)
    datas = str(datas[0])
    st = datas.split(' ')[0].strip().lower()
    if special_match(st) and st != '':
        clean_data.append(st[:20] if len(st) > 20 else st)
    
clean_data = list(dict.fromkeys(clean_data))
df = pd.DataFrame(clean_data, columns=['name'])
print (df)
df.to_csv('Name.csv')
f = open('Name.txt', 'w')
for name in clean_data:
    f.write(name + '\n')
