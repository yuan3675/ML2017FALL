import sys
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

reader = FileReader.FileReader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()

X_train, X_valid = X_train[:-20000], X_train[-20000:]
Y_train, Y_valid = Y_train[:-20000], Y_train[-20000:]

#prepare tokenizer
with open('smallTokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocabSize = 1000

#to_sequence
for i in X_valid:
    print(i)
X_train = tokenizer.texts_to_matrix(X_train, mode = 'count')
X_valid = tokenizer.texts_to_matrix(X_valid, mode = 'count')



#define the model
model = Sequential()
model.add(Dense(128, input_dim = 1000, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(Dense(64, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(Dense(32, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(Dense(16, activation = 'relu', kernel_initializer = 'glorot_normal'))
model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_normal'))
          
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('Model/Report/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

#training
history = model.fit(
    X_train, Y_train,
    batch_size=64,
    epochs=50,
    validation_data = (X_valid, Y_valid),
    callbacks = callbacks
    )

#save model
#model.save(sys.argv[3])

#save history
#with open('fulltrainHistoryDict-defaultRecurrentActive', 'wb') as file_pi:
#    pickle.dump(history.history, file_pi)


