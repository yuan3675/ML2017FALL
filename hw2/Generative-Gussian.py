import sys
import csv
import numpy as np
import CSVreader
import dataProcessing
from numpy.linalg import inv

#define parameters
outputName = sys.argv[3]
threshold = 0.5	

def sigmoid(z):
        z = -1 * z
        res = 1/(1+np.exp(z))
        #return np.clip(res, 0.0000000001, 0.9999999999)
        return res
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
		if predictValue[i] == actualValue[i]:
			counter = counter + 1
	return counter/totalNum[0]

#get data
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
X_train = reader.readX_Train(sys.argv[1])
Y_train = reader.readY_Train(sys.argv[2])
X_train = np.array(X_train).astype(float)
Y_train = np.array(Y_train).astype(float)
#X_train = process.normalize(X_train)

dim = 106
X_train_size = len(X_train)

#calculate mu1 and mu2
mu1 = np.zeros((dim,))
mu2 = np.zeros((dim,))
cnt1 = 0
cnt2 = 0
for i in range(X_train_size):
    if Y_train[i] == 1:
        mu1 += X_train[i]
        cnt1 += 1
    else:
        mu2 += X_train[i]
        cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

#calculate sigma1 and sigma2
sigma1 = np.zeros((dim, dim))
sigma2 = np.zeros((dim, dim))
for i in range(X_train_size):
    if Y_train[i] == 1:
        sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
    else:
        sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2

#calculate shared sigma
shared_sigma = (float(cnt1) / X_train_size) * sigma1 + (float(cnt2) / X_train_size) * sigma2

#calculate probability
w = np.dot(np.transpose((mu1 - mu2)), inv(shared_sigma))
b = (-1/2) * np.dot(np.dot(np.transpose(mu1), inv(shared_sigma)), mu1) + (1/2) * np.dot(np.dot(np.transpose(mu2), inv(shared_sigma)), mu2) + np.log(float(cnt1)/cnt2)
prob = sigmoid(np.dot(X_train, w) +  b)
pred = transformValue(threshold, prob)
accuracy = accuracyFunction(pred, Y_train)
print('accuracy:', accuracy)

      
"""
#store parameters
with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([b])
    for i in w:
        writer.writerow([i])
    csvfile.close()

"""
