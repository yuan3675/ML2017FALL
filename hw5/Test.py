import numpy as np
import sys
import csv
import FileReader
import keras.backend as K
from keras.models import load_model

reader = FileReader.FileReader()
test = reader.readTest(sys.argv[1])
user_id, movie_id = reader.getIDs(test)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true)**2)) 

#load_model
model = load_model('Model/best/model-00103-0.84016.h5', custom_objects={'rmse': rmse})

predictValue = model.predict([user_id, movie_id])

#save result
with open(sys.argv[2], 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    index = 1
    csvHeader = ['TestDataID','Rating']
    writer.writerow(csvHeader)
    for i in predictValue:
        writer.writerow([index, float(i)])
        #writer.writerow([int(i[0]), i[1]])
        index = index + 1
    csvfile.close()

