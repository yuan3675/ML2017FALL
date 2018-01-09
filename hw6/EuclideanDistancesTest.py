import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances
from keras import backend as K
from theano import function
import csv, sys

data = np.load('image.npy')
data = data.astype('float32') / 255

model = load_model(sys.argv[1])
model.summary()

encoded_imgs = model.predict(data)
output = []

print('start predicting......')
with open('test_case.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in reader:
        ID, test1_index, test2_index = row
        if counter != 0:
            result = euclidean_distances([encoded_imgs[int(test1_index)]], [encoded_imgs[int(test2_index)]])
            if result[0][0] <= 30:
                output.append([int(ID), 1])
            else:
                output.append([int(ID), 0])
        counter = counter + 1
csvfile.close()

#save result
print('start writting......')
with open('EDoutput.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvHeader = ['ID','Ans']
    writer.writerow(csvHeader)
    for i in output:
        writer.writerow([i[0], i[1]])
csvfile.close()

