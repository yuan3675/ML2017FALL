import sys
import csv
import CSVreader
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

def transformValue(predict):
    temp = []
    for i in predict:
        index = 0
        big = 0
        for j in range(len(i)):
            if i[j] >= big:
                big = i[j]
                index = j
        temp.append(index)
    return temp
    
reader = CSVreader.CSVreader()
test = reader.getX_Test(sys.argv[1])
test = np.array(test).astype(float)
test /= 255
model = load_model(sys.argv[2])

predictValue = model.predict(test)
predictValue = transformValue(predictValue)


with open(sys.argv[3], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 0
    csvHeader = ['id','label']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow([index, int(i)])
        index = index + 1
    csvfile.close()

