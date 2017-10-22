import sys
import csv
import CSVreader
import numpy as np
import dataProcessing

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
    
threshold = 0.5
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
test = reader.readTest(sys.argv[1])
test = np.array(test).astype(float)
#test = process.normalize(test)
w = reader.readParameters(sys.argv[2])
b = w.pop(0)
w = np.array(w).astype(float)

prob = sigmoid(np.dot(test, w) + b)
pred = transformValue(threshold, prob)

with open(sys.argv[3], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in pred:
        index = index + 1
        writer.writerow([index, int(i)])
    csvfile.close()

