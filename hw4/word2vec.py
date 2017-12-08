# -*- coding: utf-8 -*-
import sys
import FileReader
import pickle
import numpy as np
from gensim import corpora
from gensim.models import word2vec
from collections import defaultdict

def Word2Vec(data):
    #delete noisy word
    stopList = set('for a of the and to in .'.split())
    newData = []
    for sequence in data:
        newSequence = []
        words = sequence.split()
        for word in words:
            if word not in stopList:
                newSequence.append(word)
        newData.append(newSequence)

    #load word2vec model
    word2VecModel = word2vec.Word2Vec.load("allSequences.bin")

    #encode sequences
    encodedData = []
    print('Start encoding')
    for sequence in newData:
        encodedSequence = []
        for word in sequence:
            if word in word2VecModel.wv.vocab:
                encodedSequence.append(word2VecModel.wv[word])
            else:
                encodedWord = [0] * 128
                encodedSequence.append(encodedWord)
            
        encodedData.append(encodedSequence)

    #pad X_train and X_valid to  a max length of 40 words
    print('Start padding')
    maxLength = 40        
    paddedData = pad_sequences(encodedData, maxlen = maxLength, dtype = 'float64', padding = 'post')

    return (paddedData)

reader = FileReader.FileReader()
X_train = reader.getX_Train(sys.argv[2])
Y_train = reader.getY_Train()

X_train, X_valid = X_train[:-20000], X_train[-20000:]
Y_train, Y_valid = Y_train[:-20000], Y_train[-20000:]

    paddedTrain = Word2Vec(X_train)
    paddedTrain = Word2Vec(X_valid)

elif sys.argv[1] == 2:
    X_test = reader.getX_Test()
