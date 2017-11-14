import numpy as np
import pandas as pd
import sys
import CSVreader
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

#get data
print('Preprocessing......')
reader = CSVreader.CSVreader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()

X_train = np.array(X_train).astype(float)
X_train /= 255
Y_train = np.array(Y_train).astype(int)

X_train, X_valid = X_train[:-3000], X_train[-3000:]
Y_train, Y_valid = Y_train[:-3000], Y_train[-3000:]

datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)

#Construct CNN
print('Constructing CNN......')
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.4))


model.summary()

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

callbacks = []
callbacks.append(ModelCheckpoint('Model/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

#Train
history = model.fit_generator(
    datagen.flow(X_train, Y_train, batch_size=128),
    steps_per_epoch = len(X_train)//128,
    epochs = 250,
    validation_data = (X_valid, Y_valid),
    callbacks = callbacks
    )



#save model
model.save(sys.argv[2])

#save history
with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


