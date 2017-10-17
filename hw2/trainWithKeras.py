import numpy as np
import pandas as pd
import sys
import CSVreader
import dataProcessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
X_train = reader.readX_Train(sys.argv[1])
Y_train = reader.readY_Train(sys.argv[2])
X_train = np.array(X_train).astype(float)
Y_train = np.array(Y_train).astype(float)
X_train = process.normalize(X_train)

#Construct
model = Sequential()
model.add(Dense(106, input_dim=106, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')

#Train
model.fit(X_train, Y_train,
          epochs=60,
          batch_size=128)
score = model.evaluate(X_train, Y_train, batch_size=128)

#save model
model.save(sys.argv[3])
