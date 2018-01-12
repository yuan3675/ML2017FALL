import numpy as np
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from keras import backend as K
import csv, sys
import matplotlib.pyplot as plt
import adjustText

def plot(Xs, Ys, Texts, kmeans):
    for i in range(len(Xs)):
        if i == 0:
            first = kmeans[i]
        if kmeans[i] == 1:
            plt.plot(Xs[i], Ys[i], 'ro')
        else:
            plt.plot(Xs[i], Ys[i], 'bo')
        
    #texts = [plt.text(Xs[i], Ys[i], Texts[i]) for i in wrong]
    #plt.title(str(adjustText.adjust_text(texts, wrongX, wrongY, arrowprops = dict(arrowstyle = '->', color = 'black'))))
    plt.show()
"""
data = np.load('C:\\Users\\yuan\\Desktop\\visualization.npy')
data = data.astype('float32') / 255

model = load_model(sys.argv[1])
model.summary()
#encoded_imgs = model.predict(data)

encoder = K.function([model.get_layer('input_1').input], [model.get_layer('dense_1').output])
encoded_imgs = encoder([data])

print('Doing TSNE......')
encoded_imgs = TSNE(n_components=2).fit_transform(encoded_imgs[0])
encoded_imgs = np.array(encoded_imgs)


print('Saving result......')
with open('TSNE.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in encoded_imgs:
        writer.writerow([i[0], i[1]])
csvfile.close()
"""
encoded_imgs = []
with open('TSNE.csv', 'r', encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        encoded_imgs.append([float(row[0]), float(row[1])])
csvfile.close()
encoded_imgs = np.array(encoded_imgs)

kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
kmeans = kmeans.labels_
output = []
print(kmeans.sum())
"""
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
"""
text = []
for i in range(10000):
    text.append(i+1)

#visualization
plot(encoded_imgs[:, 0], encoded_imgs[:, 1], text, kmeans)
