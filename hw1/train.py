import sys
import csv
import math
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
iteration = 500000
learningRate = 0.0001
parameter = np.array([[0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01]
                      , [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.01]])

def hypoFunction(parameters, data):
    #print(data.shape, parameters.shape)
    predictValue = np.dot(data, parameters)
    return predictValue

def costFunction(predictValue, actualValue, m):
    error = (predictValue - actualValue)**2
    return np.sum(error)/(2*m)


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

#validSet = process.getDataSet(validData)
#validTargetSet = process.getTargetSet(validData)

#train
train_m = trainTargetSet.size
#valid_m = validTargetSet.size
n = 10

for i in range(iteration):
    predictValue = hypoFunction(parameter, trainSet)
    parameter = parameter - learningRate * np.dot(np.transpose(trainSet), (predictValue - trainTargetSet)) / train_m
    if i % 10000 == 0:
        errorRate = costFunction(predictValue, trainTargetSet, train_m)
        print(i,'training error:', errorRate)
        



#compute validation error rate
#predictValue = hypoFunction(parameter, validSet)
#errorRate = (costFunction(predictValue, validTargetSet))/valid_m * 100
#print('Validation error rate=', errorRate, '%')

with open('parameters.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in parameter:
        writer.writerow([i])
    csvfile.close()

#compute test
predictValue = hypoFunction(parameter, testData)
with open('testOutput.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','value']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow(['id_'+str(index), i[0]])
        index = index + 1
    csvfile.close()


