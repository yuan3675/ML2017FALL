import sys
import csv


class CSVreader:
    def __init__(self):
        self.trainList = []
        self.testList = []
        self.parameterList = []
        
    def readX_Train(self, filename):
        temp = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 1:
                    break
                temp.append(row)
        csvfile.close()
        temp.pop(0)
        return(temp)

    def readY_Train(self, filename):
        temp = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                temp.append(row)
        csvfile.close()
        temp.pop(0)
        return(temp)
    
    def readTest(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.testList.append(row)
        csvfile.close()
        return(self.testList)

    def readParameters(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.parameterList.append(float(row[0]))
        csvfile.close()
        return(self.parameterList)



