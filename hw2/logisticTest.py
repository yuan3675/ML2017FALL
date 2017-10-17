import sys
import csv
import CSVreader
import dataProcessing
import pandas as pd
import numpy as np

def sigmoid(z):
        z = -1 * z
        return 1/(1+np.exp(z))
	
def hypoFunction(weights, data):
        z = np.dot(data, weights) 
        return sigmoid(z)

reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
parameter = np.array(reader.readParameters('parameter_logistic_first_try.csv')).astype(float)

test = reader.readX_Train(sys.argv[1])
test = np.array(test).astype(float)
test = process.normalize(test)

outputName = sys.argv[2]

predictValue = hypoFunction(parameter, test)

with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow([index, i])
        index = index + 1
    csvfile.close()

