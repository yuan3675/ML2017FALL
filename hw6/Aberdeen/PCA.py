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
    if i == 0 or i == 10 or i == 20 or i == 30:
        target.append(new_img)

data = np.array(data)
mean = np.mean(data, axis = 0)

for i in range(len(data)):
    data[i] = data[i] - mean

trans_data = np.transpose(data)


U, S, V = np.linalg.svd(trans_data, full_matrices=False)

pics = []

#calculate the weights of 4 eigenfaces
for i in range(4):
    print(S[i] / np.sum(S))

"""
#draw 4 eigenfaces
for i in range(4):
    M = U[:, i]
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    pics.append(M)
"""


for j in range(4):
    weights = []
    for i in range(4):
        weights.append(np.dot(data[random.randint(0, 414)], U[:, i]))
   
    M = [0] * 270000
    M = np.array(M)
    for i in range(4):
        M = M + weights[i] * np.transpose(U[:, i])
    M = M + mean

    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    pics.append(M)

plt.figure(figsize=(16,4))
#display resize
for i in range(4):
    ax = plt.subplot(1, 4, i+1)
    plt.imshow(pics[i].reshape(300, 300, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

"""

ax = plt.imshow(pics[0].reshape(300,300,3))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
"""
plt.show()
#plt.imsave('4eigenface.jpg')

