import numpy as np
import pandas as pd
import sys
import CSVreader
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model

#get data
print('Preprocessing......')
reader = CSVreader.CSVreader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()
X_test = reader.getX_Test(sys.argv[2])
X_train = np.array(X_train).astype(int)
Y_train = np.array(Y_train).astype(int)
#X_test = np.array(X_test).astype(int)
#print(X_test[0])


#Construct CNN
print('Constructing NN......')
model = Sequential()
model.add(Conv2D(25, (3, 3), input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(50, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

"""
#k-fold validation
total_loss = 0
for i in range(10):
    x_train = process.getTrainSet(X_train, 10, i)
    x_valid = process.getValidSet(X_train, 10, i)
        
    y_train = process.getTrainSet(Y_train, 10, i)
    y_valid = process.getValidSet(Y_train, 10, i)

    model.fit(x_train, y_train,
          epochs=1,
          batch_size=128, validation_data=(x_valid, y_valid))
    loss = model.evaluate(x_valid, y_valid, batch_size=128)
    total_loss = total_loss + loss

print('\naverage loss:', total_loss/10)
"""

#Train
model.fit(X_train, Y_train,
          epochs=20,
          batch_size=100)
score = model.evaluate(X_train, Y_train)
print('Train Accuracy:', score)


#save model
model.save(sys.argv[3])

