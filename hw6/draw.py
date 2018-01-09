import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.models import load_model

data = np.load('image.npy')
#data = data.reshape((140000, 28, 28, 1))

model = load_model(sys.argv[1])

decoded_imgs = model.predict(data)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
