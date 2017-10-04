import sys
import csv

class CSVreader:
    def __init__(self):
        self.trainList = []
        self.testList = []
        
    def readTrain(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                del row[1]
                if row[1] == 'PM2.5':
                    self.trainList.append(row)
            return(self.trainList)

    def readTest(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if row[1] == 'PM2.5':
                    self.testList.append(row)
            return(self.testList)


