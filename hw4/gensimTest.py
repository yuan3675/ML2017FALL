import sys
import csv
import FileReader
import pickle
import numpy as np
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from gensim import corpora
from gensim.models import word2vec

def transformValue(predict):
    temp = []
    threshold = 0.5
    for i in predict:
        if i[0] > threshold:
            i[0] = 1
        else:
            i[0] = 0
        temp.append(i)
    return temp

def highPercentageTransform(trainingData, predictLabel):
    temp = []
    threshold = 0.7
    index = 0
    for i in predictLabel:
        if i[0] >= threshold:
            i[0] = 1
            line = [i[0], trainingData[index]]
            temp.append(line)
        elif i[0] <= (1 - threshold):
            i[0] = 0
            line = [i[0], trainingData[index]]
            temp.append(line)
        index = index + 1
    return temp

#read data
reader = FileReader.FileReader()
X_test = reader.getX_Test(sys.argv[1])

#load model
model = load_model('Model\model-00031-0.82305.h5')

#delete noisy word
stopList = set('for a of the and to in .'.split())
newX_test = []
for sequence in X_test:
    newSequence = []
    words = sequence.split()
    for word in words:
        if word not in stopList:
            newSequence.append(word)
    newX_test.append(newSequence)

#load word2vec model
word2VecModel = word2vec.Word2Vec.load("allSequences.bin")

#encode sequences
encodedTest = []
for sequence in newX_test:
    encodedSequence = []
    for word in sequence:
        if word in word2VecModel.wv.vocab:
            encodedSequence.append(word2VecModel.wv[word])
        else:
            encodedWord = []
            for i in range(128):
                encodedWord.append(0)
            encodedSequence.append(encodedWord)
            
    encodedTest.append(encodedSequence)

#pad X_train and X_valid to  a max length of 40 words
maxLength = 40        
paddedTest = pad_sequences(encodedTest, maxlen = maxLength, dtype = 'float64', padding = 'post')

#predict test data
predictValue = model.predict(paddedTest)
predictValue = transformValue(predictValue)
#predictValue = highPercentageTransform(X_test, predictValue)

#save result
with open(sys.argv[2], 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow([index, int(i)])
        #writer.writerow([int(i[0]), i[1]])
        index = index + 1
    csvfile.close()

