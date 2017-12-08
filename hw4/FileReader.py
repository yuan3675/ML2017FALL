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

    def getX_Nolabel(self, filename):
        temp = []
        decrement = True
        maxInt = sys.maxsize

        #unlabel data size too large, do some processes        
        while decrement:
            # decrease the maxInt value by factor 10 
            # as long as the OverflowError occurs.

            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True

        csv.field_size_limit(maxInt)
        with open(filename, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                sentence = " ".join(row)
                temp.append(sentence)
        csvfile.close()
        return temp

    def getFullTrain(self, labeledfile, unlabeledfile, predictlabel):
        decrement = True
        maxInt = sys.maxsize

        #unlabel data size too large, do some processes        
        while decrement:
            # decrease the maxInt value by factor 10 
            # as long as the OverflowError occurs.

            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True

        csv.field_size_limit(maxInt)

        with open(labeledfile, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                label = row.pop(0)
                row.pop(0)
                sentence = " ".join(row)
                self.labels.append(label)
                self.features.append(sentence)
        csvfile.close()

        with open(unlabeledfile, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                sentence = " ".join(row)
                self.features.append(sentence)
        csvfile.close()

        with open(predictlabel, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            count = 0
            for row in reader:
                if count > 0:
                    row.pop(0)
                    self.labels.append(row.pop(0))
                count = count + 1
        csvfile.close()
        
        #print(len(self.features))
        return self.features

    def getHighPercentage(self, labeledfile, highPercentageData):
        decrement = True
        maxInt = sys.maxsize

        #unlabel data size too large, do some processes        
        while decrement:
            # decrease the maxInt value by factor 10 
            # as long as the OverflowError occurs.

            decrement = False
            try:
                csv.field_size_limit(maxInt)
            except OverflowError:
                maxInt = int(maxInt/10)
                decrement = True

        csv.field_size_limit(maxInt)
        
        with open(labeledfile, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                label = row.pop(0)
                row.pop(0)
                sentence = " ".join(row)
                self.labels.append(label)
                self.features.append(sentence)
        csvfile.close()

        with open(highPercentageData, encoding='utf-8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                label = row.pop(0)
                sentence = " ".join(row)
                self.labels.append(label)
                self.features.append(sentence)
        csvfile.close()

        return self.features
