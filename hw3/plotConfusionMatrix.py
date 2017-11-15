import sys
import csv
import CSVreader
import numpy as np
import matplotlib.pyplot as plt
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def transformValue(predict):
    temp = []
    for i in predict:
        index = 0
        big = 0
        for j in range(len(i)):
            if i[j] >= big:
                big = i[j]
                index = j
        temp.append(index)
    return temp

def plotconfusionmatrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
reader = CSVreader.CSVreader()
X_train = reader.getX_Train(sys.argv[1])
Y_train = reader.getY_Train()

X_train = np.array(X_train).astype(float)
X_train /= 255
Y_train = np.array(Y_train).astype(int)

X_train, X_valid = X_train[:-3000], X_train[-3000:]
Y_train, Y_valid = Y_train[:-3000], Y_train[-3000:]
model = load_model(sys.argv[2])

predictValue = model.predict(X_valid)
predictValue = transformValue(predictValue)
Y_valid = transformValue(Y_valid)

cnf_matrix = confusion_matrix(Y_valid, predictValue)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

plt.figure()
plotconfusionmatrix(cnf_matrix, classes=class_names)

plt.show()

