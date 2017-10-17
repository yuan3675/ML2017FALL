import sys
import csv
import math
import pandas as pd
import numpy as np
import CSVreader
import dataProcessing

#define parameters
Lambda = 0.0001
iteration = 5001
learningRate = 0.0000001
weights = np.random.random((106, 1))
outputName = sys.argv[3]
threshold = 0.5				  
bias = 0.005

def sigmoid(z):
        z = -1 * z
        return 1/(1+np.exp(z))
	
def hypoFunction(weights, data):
        z = np.dot(data, weights)
        return sigmoid(z)

def transformValue(threshold, value):
        for i in range(len(value)):
                if value[i] > threshold:
                        value[i] = 1
                else:
                        value[i] = 0
        return value

def accuracyFunction(predictValue, actualValue):
	totalNum = predictValue.shape
	counter = 0
	for i in range(totalNum[0]):
		if predictValue[i][0] == actualValue[i]:
			counter = counter + 1
	return counter/totalNum[0]

#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
X_train = reader.readX_Train(sys.argv[1])
Y_train = reader.readY_Train(sys.argv[2])
X_train = np.array(X_train).astype(float)
Y_train = np.array(Y_train).astype(float)
X_train = process.normalize(X_train)

"""
#k-fold validation
for j in range(k):
"""        

#gradient descent
for i in range(iteration):
        predictValue = hypoFunction(weights, X_train)
        predictValue = transformValue(threshold, predictValue)
        parameterL = weights
        parameterL[0][0] = 0
        weights = weights - learningRate * np.dot(np.transpose(X_train), (predictValue - Y_train))
        #weights = weights - learningRate * (np.dot(np.transpose(trainSet), (predictValue - trainTargetSet)) / train_m + (Lambda / train_m) * parameterL)
        if i % 1000 == 0:
                accuracy = accuracyFunction(predictValue, Y_train)
                print(i,'training accuracy:', accuracy)
        


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
    for i in weights:
        writer.writerow([i[0]])
    csvfile.close()

