import numpy as np
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras import backend as K
from theano import function
import csv, sys

data = np.load('image.npy')
data = data.astype('float32') / 255

#model = load_model(sys.argv[1])

#print('predicting......')
#encoded_imgs = model.predict(data)

print('Doing TSNE......')
encoded_imgs = TSNE(n_components=2).fit_transform(data)

print('writing data......')
with open('TSNEVector.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in encoded_imgs:
        writer.writerow([i[0], i[1]])
csvfile.close()
