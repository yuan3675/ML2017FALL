from sys import argv
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import CSVreader
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import load_model
from keras.utils import plot_model, np_utils
from sklearn.metrics import confusion_matrix
import keras.backend as K

CATEGORY = 7
SHAPE = 48

reader = CSVreader.CSVreader()
X_train = reader.getX_Train(argv[1])
Y_train = reader.getY_Train()

X_train = np.array(X_train).astype(float)
X_train /= 255
Y_train = np.array(Y_train).astype(int)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

print('load model......')
model_name = argv[2]
model = load_model(model_name)

label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral']

print('print sailency map......')
plt.figure(figsize=(16, 6))
emotion_classifier = load_model(model_name)
input_img = emotion_classifier.input
img_ids = [(1, 8787)]

for i, idx in img_ids:
    print('plot figure %d.' %idx)
    img = X_train[idx].reshape(1, 48, 48, 1)
    val_proba = emotion_classifier.predict(img)
    pred = val_proba.argmax(axis=-1)
    target = K.mean(emotion_classifier.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.funciton([input_img, K.learning_phase()], [grads])

    heatmap = fn([img, 0])[0]
    heatmap = heatmap.reshape(48, 48)
    heatmap /= heatmap.std()

    see = img.reshape(48, 48)
    plt.subplot(1, 3, i)
    plt.imshow(see, cmap = 'gray')
    plt.title("%d. %s" % (idx, label[Y[idx].argmax()]))

    thres = heatmap.std()
    see[np.where(abs(heatmap) <= thres)] = np.mean(see)

    plt.subplot(1, 3 , i+1)
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1, 3, i+2)
    plt.imshow(see, cmap = 'gray')
    plt.colorbar()
    plt.tight_layout()

plt.show()
