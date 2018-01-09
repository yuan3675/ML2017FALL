import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.cluster import KMeans
import csv

data = np.load('image.npy')
#normalize
data = data.astype('float32') / 255
#data = data.reshape((140000, 28, 28, 1))

X_train, X_valid = data[:-14000], data[-14000:]

encoding_dim = 16
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_valid_noisy = X_valid + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_valid.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_valid_noisy = np.clip(X_valid_noisy, 0., 1.)

"""
#Construct auto-encoder
input_img = Input(shape=(28, 28, 1))
#encoded layer
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)

#decoded layer
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
"""
#construct DNN auto-encoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
#encoded = Dense(100, activation='relu')(encoded)
#encoded = Dense(encoding_dim, activation='relu')(encoded)

#decoded = Dense(64, activation='relu')(encoded)
#decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
#decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

#autoencoder model
autoencoder = Model(input_img, decoded)
#encoder model
encoder = Model(input_img, encoded)
#encoded_input = Input(shape=(encoding_dim,))
#decoder_layer = autoencoder.layers[-4:]

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

callbacks = []
callbacks.append(ModelCheckpoint('Model\model-{epoch:05d}-{val_loss:.5f}.h5', monitor='val_loss', mode = 'min', save_best_only=True, period=1))

autoencoder.fit(X_train, X_train,
                epochs = 2000,
                batch_size = 256,
                shuffle = True,
                validation_data = (X_valid, X_valid),
                callbacks = callbacks)

autoencoder.save('2000Autoencoder.h5')
encoder.save('DNNencoder.h5', overwrite = True)
