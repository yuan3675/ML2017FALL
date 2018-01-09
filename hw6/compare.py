import numpy as np
import sys
from keras.models import load_model
from sklearn.cluster import KMeans
from keras import backend as K

data = np.load('image.npy')
data = data.astype('float32') / 255

model1 = load_model(sys.argv[1])
model2 = load_model(sys.argv[2])

encoder1 = K.function([model1.get_layer('input_1').input], [model1.get_layer('dense_2').output])
encoder2 = K.function([model2.get_layer('input_1').input], [model2.get_layer('dense_3').output])
encoded_img1 = encoder1([data])
encoded_img2 = encoder2([data])

kmeans1 = KMeans(n_clusters=2, random_state=0).fit(encoded_img1[0])
kmeans1 = kmeans1.labels_

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(encoded_img2[0])
kmeans2 = kmeans2.labels_

counter = 0
for i in range(len(kmeans1)):
    if kmeans1[i] != kmeans2[i]:
        counter = counter + 1

print('there are', counter, 'diferences.')
