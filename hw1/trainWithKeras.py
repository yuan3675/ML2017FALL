import sys
import csv
import math
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#initialization
model = Sequential()
model.add(Dense(19, input_dim=9, activation='linear', use_bias=True))
model.add(Dense(1, activation='linear', use_bias=True))

#choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd', metrics = ['accuracy'])

#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
data = pd.DataFrame(reader.readTrain(sys.argv[1]))
test = pd.DataFrame(reader.readTest(sys.argv[2]))


trainData = process.getTrain(data)
#validData = process.getValid(data)
testData = process.getTest(test)

trainSet = process.getDataSet(trainData)
trainTargetSet = process.getTargetSet(trainData)


#training
print('Training.......')
model.fit(trainSet, trainTargetSet, epochs=250, batch_size=2)
score = model.evaluate(trainSet, trainTargetSet, batch_size=2)

#weight, b = model.layers[0].get_weights()
"""
with open('parameters_NN.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in weight:
        writer.writerow([i])
    csvfile.close()
"""
#print(weight)
#predict
classes = model.predict(testData, batch_size=2)
print(classes)

with open('testOutput_NN.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','value']
    writer.writerow(csvHeader)
    for i in classes:
        writer.writerow(['id_'+str(index), i[0]])
        index = index + 1
    csvfile.close()


