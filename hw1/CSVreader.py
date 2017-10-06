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
                del row[0]
                del row[0]
                if row[0]=='PM2.5':
                    del row[0]
                    self.trainList.append(row)
            return(self.trainList)
        
    def readTest(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                del row[0]
                if row[0]=='PM2.5':
                    del row[0]
                    self.testList.append(row)
            return(self.testList)




