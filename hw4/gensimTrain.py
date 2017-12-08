import sys
import FileReader
import pickle
import numpy as np
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LSTM, Masking, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from gensim import corpora
from gensim.models import word2vec
from collections import defaultdict

reader = FileReader.FileReader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()

X_train, X_valid = X_train[:-20000], X_train[-20000:]
Y_train, Y_valid = Y_train[:-20000], Y_train[-20000:]

#delete noisy word
stopList = set('for a of the and to in .'.split())
newX_train = []
newX_valid = []
for sequence in X_train:
    newSequence = []
    words = sequence.split()
    for word in words:
        if word not in stopList:
            newSequence.append(word)
    newX_train.append(newSequence)

for sequence in X_valid:
    newSequence = []
    words = sequence.split()
    for word in words:
        if word not in stopList:
            newSequence.append(word)
    newX_valid.append(newSequence)


#load word2vec model
word2VecModel = word2vec.Word2Vec.load("allSequences.bin")

#encode sequences
#print('start padding')
encodedTrain = []
encodedValid = []
for sequence in newX_train:
    encodedSequence = []
    for word in sequence:
        if word in word2VecModel.wv.vocab:
            encodedSequence.append(word2VecModel.wv[word])
        else:
            encodedWord = []
            for i in range(128):
                encodedWord.append(0)
            encodedSequence.append(encodedWord)
            
    encodedTrain.append(encodedSequence)

for sequence in newX_valid:
    encodedSequence = []
    for word in sequence:
        if word in word2VecModel.wv.vocab:
            encodedSequence.append(word2VecModel.wv[word])
        else:
            encodedWord = [0] *128
            encodedSequence.append(encodedWord)
            
    encodedValid.append(encodedSequence)

#pad X_train and X_valid to  a max length of 40 words
maxLength = 40        
paddedTrain = pad_sequences(encodedTrain, maxlen = maxLength, dtype = 'float64', padding = 'post')
paddedValid = pad_sequences(encodedValid, maxlen = maxLength, dtype = 'float64', padding = 'post')


#define the model
model = Sequential()
model.add(Masking(mask_value=0, input_shape = (40, 128)))
model.add(Bidirectional(LSTM(128, activation = 'sigmoid',
               dropout=0.3, recurrent_dropout=0.3,
               kernel_initializer = 'glorot_normal',
               return_sequences = True)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, activation = 'sigmoid',
               dropout=0.5, recurrent_dropout=0.5,
               kernel_initializer = 'glorot_normal')))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_normal'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('Model/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

#training
print('Start training')
history = model.fit(
    paddedTrain, Y_train,
    batch_size=64,
    epochs=50,
    validation_data = (paddedValid, Y_valid),
    callbacks = callbacks
    )

#save model
#model.save(sys.argv[3])

#save history
with open('gensimMaskBiHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

