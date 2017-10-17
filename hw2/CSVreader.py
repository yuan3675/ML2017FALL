import sys
import csv


class CSVreader:
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
        temp = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) != 106:
                    break
                temp.append(row)
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



