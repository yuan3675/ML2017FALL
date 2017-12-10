import sys
import FileReader
import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

reader = FileReader.FileReader()
userID = reader.getUserID('userID.txt')
movieID = reader.getMovieID('movieID.txt')
targetMatrix = reader.getMatrix('matrix.txt')

userID = np.array(userID)
movieID = np.array(movieID)
targetMatrix = np.array(targetMatrix)

n_users = np.max(userID) + 1
n_movies = np.max(movieID) + 1

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true)**2)) 

#input
input_userID = keras.layers.Input(shape=(1,))
input_movieID = keras.layers.Input(shape=(1,))
#embedding input
embed_userID = keras.layers.Embedding(n_users, 128)(input_userID)
embed_movieID = keras.layers.Embedding(n_movies, 128)(input_movieID)
vector_userID = keras.layers.Flatten()(embed_userID)
vector_movieID = keras.layers.Flatten()(embed_movieID)
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
model = keras.models.Model(inputs = [input_userID, input_movieID], outputs=out)
model.summary()

model.compile(loss = 'mse',
              optimizer='adam',
              metrics = [rmse])

callbacks = []
callbacks.append(ModelCheckpoint('model-{epoch:05d}-{val_rmse:.5f}.h5', monitor='val_rmse', mode = 'min', save_best_only=True, period=1))

history = model.fit(
    [userID, movieID], targetMatrix,
    epochs = 1000,
    batch_size = 10000,
    validation_split = 0.1,
    callbacks = callbacks
    )



