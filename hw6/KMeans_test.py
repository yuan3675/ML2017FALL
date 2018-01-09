import numpy as np
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras import backend as K
import csv, sys

data = np.load('image.npy')
data = data.astype('float32') / 255

model = load_model(sys.argv[1])
model.summary()
#encoded_imgs = model.predict(data)

encoder = K.function([model.get_layer('input_1').input], [model.get_layer('dense_1').output])
encoded_imgs = encoder([data])

"""
encoded_imgs = []
with open('TSNEVector.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        vec1, vec2 = row
        encoded_imgs.append([float(vec1), float(vec2)])
csvfile.close()
"""

kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs[0])
kmeans = kmeans.labels_
output = []
print(kmeans.sum())

print('start predicting......')
with open('test_case.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in reader:
        ID, test1_index, test2_index = row
        if counter != 0:
            if kmeans[int(test1_index)] == kmeans[int(test2_index)]:
                output.append([int(ID), 1])
            else:
                output.append([int(ID), 0])
        counter = counter + 1
csvfile.close()

#save result
print('start writting......')
with open('KMeansoutput.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvHeader = ['ID','Ans']
    writer.writerow(csvHeader)
    for i in output:
        writer.writerow([i[0], i[1]])
csvfile.close()

