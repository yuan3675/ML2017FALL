import sys
import csv
import CSVreader
import numpy as np
import matplotlib.pyplot as plt
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

def plotConfusionMatrix(cm, classes, title='Confusion Matrix', camp=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', camp=camp)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
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

cnf_matrix = confusion_matrix(Y_valid, predictValue)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = (0, 1, 2, 3, 4, 5, 6)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion Matrix')

plt.show()
