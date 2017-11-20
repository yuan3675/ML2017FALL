import sys
import csv

class FileReader:
    features = []
    labels = []
    trainingDataNum = -1
    
    def getX_Train(self, filename):
        with open(filename, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                label = row.pop(0)
                row.pop(0)
                sentence = " ".join(row)
                self.labels.append(label)
                self.features.append(sentence)
        csvfile.close()
        return(self.features)

    def getY_Train(self):
        return(self.labels)
    
    def getX_Test(self, filename):
        temp = []
        with open(filename, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                row.pop(0)
                sentence = " ".join(row)
                temp.append(sentence)
        csvfile.close()
        temp.pop(0)
        return(temp)



