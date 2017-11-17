import sys
import csv

class CSVreader:
    features = []
    labels = []
    trainingDataNum = -1
    
    def getX_Train(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if self.trainingDataNum != -1:
                    self.labels.append(row[0])
                    row.pop(0)
                    splitRow = row[0].split()
                    splitRow = [splitRow[i:i+1] for i in range(0, len(splitRow), 1)]
                    splitRow = [splitRow[i:i+48] for i in range(0, len(splitRow), 48)]
                    self.features.append(splitRow)
                self.trainingDataNum += 1
        csvfile.close()
        return(self.features)
    
    def getDNNX_Train(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if self.trainingDataNum != -1:
                    self.labels.append(row[0])
                    row.pop(0)
                    splitRow = row[0].split()
                    splitRow = [splitRow[i:i+48] for i in range(0, len(splitRow), 48)]
                    self.features.append(splitRow)
                self.trainingDataNum += 1
        csvfile.close()
        return(self.features)

    def getY_Train(self):
        for i in range(len(self.labels)):
            if self.labels[i] == '0':
                self.labels[i] = [1, 0, 0, 0, 0, 0, 0]
            elif self.labels[i] == '1':
                self.labels[i] = [0, 1, 0, 0, 0, 0, 0]
            elif self.labels[i] == '2':
                self.labels[i] = [0, 0, 1, 0, 0, 0, 0]
            elif self.labels[i] == '3':
                self.labels[i] = [0, 0, 0, 1, 0, 0, 0]
            elif self.labels[i] == '4':
                self.labels[i] = [0, 0, 0, 0, 1, 0, 0]
            elif self.labels[i] == '5':
                self.labels[i] = [0, 0, 0, 0, 0, 1, 0]
            else:
                self.labels[i] = [0, 0, 0, 0, 0, 0, 1]
        return(self.labels)
    
    def getX_Test(self, filename):
        temp = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                row.pop(0)
                splitRow = row[0].split()
                sr = []
                for i in splitRow:
                    sr.append([i])
                splitRow = [sr[i:i+48] for i in range(0, len(sr), 48)]
                temp.append(splitRow)
        csvfile.close()
        temp.pop(0)
        return(temp)

    def readParameters(self, filename):
        temp = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                temp.append(float(row[0]))
        csvfile.close()
        return(temp)



