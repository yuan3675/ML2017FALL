import sys
import csv
import math
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
Lambda = 0.0001
iteration = 100001
learningRate = 0.000003
parameter = np.array([[0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01],
                      [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]).astype(float)

def hypoFunction(parameters, data):
    predictValue = np.dot(data, parameters)
    return predictValue

def costFunction(predictValue, actualValue, m, para):
    error = (predictValue - actualValue)**2
    reg = np.sum(para**2) * Lambda
    return (np.sum(error) + reg)/(2*m)


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

print('trainSet size:', trainSet.shape)
print('targetSet size', trainTargetSet.shape)

for i in range(iteration):
    predictValue = hypoFunction(parameter, trainSet)
    parameterL = parameter
    parameterL[0][0] = 0
    parameter = parameter - learningRate * (np.dot(np.transpose(trainSet), (predictValue - trainTargetSet)) / train_m + (Lambda / train_m) * parameterL)
    if i % 10000 == 0:
        errorRate = costFunction(predictValue, trainTargetSet, train_m, parameter)
        print(i,'training error:', errorRate)
        


"""
#compute validation error rate
predictValue = hypoFunction(parameter, validSet)
errorRate = costFunction(predictValue, validTargetSet, valid_m)
print('Validation error =', errorRate)
"""
print(parameter.shape)
#store parameters
with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in parameter:
        writer.writerow([i[0]])
    csvfile.close()

