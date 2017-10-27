import sys
import csv
import CSVreader
import numpy as np
import dataProcessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

def transformValue(threshold, value):
        for i in range(len(value)):
                if value[i][0] > threshold:
                        value[i][0] = 1
                else:
                        value[i][0] = 0
        return value
    
threshold = 0.5
reader = CSVreader.CSVreader()
process = dataProcessing.dataProcessing()
test = reader.readTest(sys.argv[5])
test = np.array(test).astype(float)
test = process.normalize(test)
model = load_model("model_keras_first_try.h5")

predictValue = model.predict(test, batch_size=128)
predictValue = transformValue(threshold, predictValue)

with open(sys.argv[6], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in predictValue:
        index = index + 1
        writer.writerow([index, int(i[0])])
    csvfile.close()

