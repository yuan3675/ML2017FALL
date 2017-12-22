import sys
import FileReader
import numpy as np
from numpy import linalg as LA
import keras
import pickle
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding

reader = FileReader.FileReader()
train = reader.readTrain(sys.argv[1])
userID, movieID, rating = reader.getIDandRating(train)
ID = reader.getID(train)

n_users = np.max(userID) + 1
n_movies = np.max(movieID) + 1

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true)**2)) 



#model
model = Sequential()
model.add(Embedding(n_users, 200, input_length = 2))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))
model.summary()

model.compile(loss = 'mse',
              optimizer='adam',
              metrics = [rmse])

callbacks = []
callbacks.append(ModelCheckpoint('Model\model-{epoch:05d}-{val_rmse:.5f}.h5', monitor='val_rmse', mode = 'min', save_best_only=True, period=1))

history = model.fit(
    ID, rating,
    epochs = 300,
    batch_size = 10000,
    validation_split = 0.05,
    callbacks = callbacks
    )
"""
#save history
with open('reportQ1History', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
"""

