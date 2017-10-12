import sys
import csv
import math
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
iteration = 100001
learningRate = 0.000001
parameter = np.array([[0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01]])

def hypoFunction(parameters, data):
    predictValue = np.dot(data, parameters)
    return predictValue

def costFunction(predictValue, actualValue, m):
    error = (predictValue - actualValue)**2
    return np.sum(error)/(2*m)


#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
data = pd.DataFrame(reader.readTrain(sys.argv[1]))
outputName = sys.argv[2]


trainData = process.getTrain(data)
#validData = process.getValid(data)

trainSet = process.getDataSet(trainData)
trainTargetSet = process.getTargetSet(trainData)

#validSet = process.getDataSet(validData)
#validTargetSet = process.getTargetSet(validData)

#train
train_m = trainTargetSet.size
#valid_m = validTargetSet.size


for i in range(iteration):
    predictValue = hypoFunction(parameter, trainSet)
    parameter = parameter - learningRate * np.dot(np.transpose(trainSet), (predictValue - trainTargetSet)) / train_m
    if i % 10000 == 0:
        errorRate = costFunction(predictValue, trainTargetSet, train_m)
        print(i,'training error:', errorRate)
        


"""
#compute validation error rate
predictValue = hypoFunction(parameter, validSet)
errorRate = costFunction(predictValue, validTargetSet, valid_m)
print('Validation error =', errorRate)
"""

#store parameters
with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in parameter:
        writer.writerow([i[0]])
    csvfile.close()

