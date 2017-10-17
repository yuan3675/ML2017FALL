import sys
import csv
import CSVreader
import dataProcessing
import pandas as pd
import numpy as np

threshold = 0.5

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

reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
weights = np.array(reader.readParameters('parameter_logistic_first_try.csv')).astype(float)

test = reader.readTest(sys.argv[1])
test = np.array(test).astype(float)
test = process.normalize(test)
outputName = sys.argv[2]

predictValue = hypoFunction(weights, test)
predictValue = transformValue(threshold, predictValue)

with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in predictValue:
        index = index + 1
        writer.writerow([index, int(i)])
    csvfile.close()

