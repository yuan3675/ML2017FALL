import sys
import csv
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
    return np.sum(error)


#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
data = pd.DataFrame(reader.readTrain(sys.argv[1]))
test = pd.DataFrame(reader.readTest(sys.argv[2]))
"""
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
    #errorRate = (costFunction(predictValue, trainTargetSet))/train_m * 50
    parameter = parameter - learningRate * np.dot(np.transpose(trainSet), (predictValue - trainTargetSet))
    if(i % 500 == 0):
        errorRate = (costFunction(predictValue, trainTargetSet))/train_m * 50
        print(i,'training error rate:', errorRate, '%')



#compute validation error rate
predictValue = hypoFunction(parameter, validSet)
errorRate = (costFunction(predictValue, validTargetSet))/valid_m * 100
print('Validation error rate=', errorRate, '%')

#print(parameter)

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

"""
