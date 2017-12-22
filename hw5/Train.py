import sys
import FileReader
import numpy as np
from numpy import linalg as LA
import keras
import pickle
import keras.backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

reader = FileReader.FileReader()
train = reader.readTrain(sys.argv[1])
userID, movieID, rating = reader.getIDandRating(train)

n_users = np.max(userID) + 1
n_movies = np.max(movieID) + 1

mean = np.mean(rating)
std = np.std(rating)

def normalize(nparray):
    for i in range(len(nparray)):
        nparray[i] = (nparray[i] - mean) / std
    return nparray

norm_rating = normalize(rating)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true)**2)) 

#input
input_userID = keras.layers.Input(shape=(1,))
input_movieID = keras.layers.Input(shape=(1,))

#embedding input
embed_userID = keras.layers.Embedding(n_users, 200)(input_userID)
embed_movieID = keras.layers.Embedding(n_movies, 200)(input_movieID)
vector_userID = keras.layers.Dropout(0.5)(keras.layers.Flatten()(embed_userID))
vector_movieID = keras.layers.Dropout(0.5)(keras.layers.Flatten()(embed_movieID))

#dot user and movie
dot = keras.layers.Dot(axes=1)([vector_userID, vector_movieID])

#bias
embed_userID_bias = keras.layers.Embedding(n_users, 1)(input_userID)
embed_movieID_bias = keras.layers.Embedding(n_movies, 1)(input_movieID)
bias_userID = keras.layers.Flatten()(embed_userID_bias)
bias_movieID = keras.layers.Flatten()(embed_movieID_bias)

#output
out = keras.layers.Add()([bias_userID, bias_movieID, dot])

#model
model = keras.models.Model(inputs = [input_userID, input_movieID], outputs=dot)
model.summary()

model.compile(loss = 'mse',
              optimizer='adam',
              metrics = [rmse])

callbacks = []
callbacks.append(ModelCheckpoint('Model\model-{epoch:05d}-{val_rmse:.5f}.h5', monitor='val_rmse', mode = 'min', save_best_only=True, period=1))

history = model.fit(
    [userID, movieID], norm_rating,
    epochs = 300,
    batch_size = 10000,
    validation_split = 0.05,
    callbacks = callbacks
    )

#save history
with open('reportQ1History', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


