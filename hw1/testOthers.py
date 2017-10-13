import sys
import csv
import CSVreader
import dataProcessing
import pandas as pd
import numpy as np

def hypoFunction(parameters, data):
    predictValue = np.dot(data, parameters)
    return predictValue

reader = CSVreader.CSVreader()
processer = dataProcessing.dataProcessing()
parameter = np.array(reader.readParameters('parameters-full_features_5hours_reg04.csv')).astype(float)

test = pd.DataFrame(reader.readTest(sys.argv[1]))
testData = processer.getTestOthers(test)
outputName = sys.argv[2]

predictValue = hypoFunction(parameter, testData)

with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','value']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow(['id_'+str(index), i])
        index = index + 1
    csvfile.close()

