import sys
import FileReader
import pickle
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

#load tokenizer
with open('punctuationTokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocabSize = 10000

#integer encode X_train and X_valid
encodedTrain = tokenizer.texts_to_sequences(X_train)
encodedValid = tokenizer.texts_to_sequences(X_valid)

#pad X_train and X_valid to  a max length of 40 words
maxLength = 40
paddedTrain = pad_sequences(encodedTrain, maxlen = maxLength, padding = 'post')
paddedValid = pad_sequences(encodedValid, maxlen = maxLength, padding = 'post')

#define the model
model = Sequential()
model.add(Embedding(vocabSize, 100))
model.add(LSTM(128, activation = 'sigmoid',
               dropout=0.3, recurrent_dropout=0.3,
               kernel_initializer = 'glorot_normal',
               return_sequences = True))
model.add(BatchNormalization())
model.add(LSTM(64, activation = 'sigmoid',
               dropout=0.5, recurrent_dropout=0.5,
               kernel_initializer = 'glorot_normal'))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_normal'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('Model/Report/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

#training
history = model.fit(
    paddedTrain, Y_train,
    batch_size=64,
    epochs=10,
    validation_data = (paddedValid, Y_valid),
    callbacks = callbacks
    )

#save model
#model.save(sys.argv[3])
"""
#save history
with open('fulltrainHistoryDict-defaultRecurrentActive', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
"""

