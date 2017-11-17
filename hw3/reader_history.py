import sys
import _pickle as cPickle
#from config import HISTORY_SAVE_DIR, VERSION_NAME
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fr = open(sys.argv[1], 'rb')
    history_ORI = cPickle.load(fr)


    plt.plot(history_ORI['acc'], label='training_data')
    plt.plot(history_ORI['val_acc'] , label='validation_data')
    plt.ylabel('acc')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.2), loc=1)
    plt.show()
