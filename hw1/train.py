import sys
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
iteration = 10000
learningRate = 0.001
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

"""    
def gradientDescent(parameters, m):
        parameters = parameters - (learningRate / m)
"""
#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
data = pd.DataFrame(reader.readTrain(sys.argv[1]))
trainData = process.getTrain(data)
validData = process.getValid(data)
trainSet = trainData.iloc[:, :9]
trainSet = process.addone(trainSet).values
trainTargetSet = trainData.iloc[:, 9:].values

#train
#for i in iteration:
trainSet = np.array(trainSet)
trainTargetSet = np.array(trainTargetSet)
predictValue = hypoFunction(parameter, trainSet.astype(float))
print(costFunction(predictValue, trainTargetSet.astype(float)))
    
