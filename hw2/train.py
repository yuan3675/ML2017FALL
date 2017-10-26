import sys
import csv
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
bias_v = 0.005

def sigmoid(z):
        z = -1 * z
        res = 1/(1+np.exp(z))
        return np.clip(res, 0.0000000001, 0.9999999999)
	
def hypoFunction(weights, data):
        z = np.dot(data, weights) + bias
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
total_accuracy = 0
for i in range(10):
        weights_v = np.random.random((106, 1))
        x_train = process.getTrainSet(X_train, 10, i)
        x_valid = process.getValidSet(X_train, 10, i)
        
        y_train = process.getTrainSet(Y_train, 10, i)
        y_valid = process.getValidSet(Y_train, 10, i)
        
        for j in range(iteration):
            predictValue = hypoFunction(weights_v, x_train)
            #parameterL = weights_v
            #parameterL[0][0] = 0
            #weights_v = weights_v - learningRate * np.dot(np.transpose(x_train), (predictValue - y_train))
            bias_v = bias_v - learningRate * np.sum(predictValue - y_train)
            weights_v = weights_v - learningRate * (np.dot(np.transpose(x_train), (predictValue - y_train)) + Lambda * weights_v)
            if j % 1000 == 0:
                predictValue_t = transformValue(threshold, predictValue)
                accuracy = accuracyFunction(predictValue_t, y_train)
                print(j, 'training accuracy:', accuracy)
        validValue = hypoFunction(weights_v, x_valid)
        validValue_t = transformValue(threshold, validValue)
        accuracy_v = accuracyFunction(validValue_t, y_valid)
        print(i, 'validation accuracy:', accuracy_v)
        print('===========================')
        total_accuracy = total_accuracy + accuracy_v

print('average accuracy:', total_accuracy / 10)
"""     

#train with all data
for i in range(iteration):
        predictValue = hypoFunction(weights, X_train)
        #parameterL = weights
        #parameterL[0][0] = 0
        #weights = weights - learningRate * np.dot(np.transpose(X_train), (predictValue - Y_train))
        bias = bias - learningRate * np.sum(predictValue - Y_train)
        weights = weights - learningRate * (np.dot(np.transpose(X_train), (predictValue - Y_train)) + Lambda * weights)
        if i % 1000 == 0:
                predictValue_t = transformValue(threshold, predictValue)
                accuracy = accuracyFunction(predictValue_t, Y_train)
                print(i,'training accuracy:', accuracy)       

#store parameters
with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([bias])
    for i in weights:
        writer.writerow([i[0]])
    csvfile.close()

