import numpy as np
import pandas as pd
import sys
import CSVreader
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

#get data
print('Preprocessing......')
reader = CSVreader.CSVreader()
X_train = reader.getDNNX_Train(sys.argv[1])
Y_train = reader.getY_Train()

X_train = np.array(X_train).astype(float)
X_train /= 255
Y_train = np.array(Y_train).astype(int)

X_train, X_valid = X_train[:-3000], X_train[-3000:]
Y_train, Y_valid = Y_train[:-3000], Y_train[-3000:]

#Construct
model = Sequential()
model.add(Dense(1024, input_dim=48*48, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

callbacks = []
callbacks.append(ModelCheckpoint('Model/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

#Train
history = model.fit(
    X_train, Y_train, batch_size=128,
    epochs = 250,
    validation_data = (X_valid, Y_valid),
    callbacks = callbacks
    )

#save model
model.save(sys.argv[2])

#save history
with open('DNNtrainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
