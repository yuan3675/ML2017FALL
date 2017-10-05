import sys
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
iteration = 10000
learningRate = 0.000001
parameter = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])

def hypoFunction(parameters, data):
    predictValue = np.dot(data, parameters)
    return predictValue

def costFunction(predictValue, actualValue):
    error = (predictValue - actualValue)**2
    result = 0
    m = 0
    for i  in error:
        result = result + i
        m = m + 1
    return result/(2*m)


#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
data = pd.DataFrame(reader.readTrain(sys.argv[1]))
trainData = process.getTrain(data)
validData = process.getValid(data)
trainSet = trainData.iloc[:, :9]
trainSet = process.addone(trainSet).values
trainTargetSet = trainData.iloc[:, 9:].values
trainSet = np.array(trainSet).astype(float)
trainTargetSet = np.array(trainTargetSet).astype(float)
validSet = validData.iloc[:, :9]
validSet = process.addone(validSet).values
validTargetSet = validData.iloc[:, 9:].values
validSet = np.array(validSet).astype(float)
validTargetSet = np.array(validTargetSet).astype(float)

#train
m = trainTargetSet.size
n = 10
for i in range(iteration):
    predictValue = hypoFunction(parameter, trainSet)
    costFunction(predictValue, trainTargetSet.astype(float))
    parameter = parameter - learningRate * np.dot(np.transpose(trainSet), (predictValue - trainTargetSet))
    if(i % 100 == 0):
        print(i,'round')

#compute training error
predictValue = hypoFunction(parameter, validSet)
for i in predictValue:
    print(i)
#compute validation error
    
