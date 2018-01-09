from skimage import io
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import sys, random


data = []
target = []
for i in range(415):
    img = io.imread(str(i) + '.jpg')
    new_img = transform.resize(img, (300 , 300, 3))
    new_img = np.array(new_img)
    new_img = new_img.flatten()
    data.append(new_img)
    if i == 10:
        target.append(new_img)

data = np.array(data)
mean = np.mean(data)

mean_array = [mean] * 270000
mean_array = np.array(mean_array)

for i in range(len(data)):
    data[i] = np.subtract(data[i], mean_array)

trans_data = np.transpose(data)

U, S, V = np.linalg.svd(trans_data, full_matrices=False)

pics = []
for j in range(4):
    weights = []
    for i in range(415):
        weights.append(np.dot(data[random.randrange(415)], U[:, i]))

    M = [0] * 270000
    M = np.array(M)
    for i in range(4):
        M = M + weights[i] * np.transpose(U[:, i])
    M = M + mean_array

    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    pics.append(M)

#display resize
for i in range(4):
    ax = plt.subplot(1, 4, i+1)
    plt.imshow(pics[i].reshape(300, 300, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
#plt.imsave('4eigenface.jpg')

