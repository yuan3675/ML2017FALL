import pandas as pd
import numpy as np
import math

class dataProcessing:
    def normalize(self, data):
        for i in range(106):
            std = np.std(data[:,i])
            mean = np.mean(data[:,i])
            #print('column',i,'std:',std,'mean:',mean)
            if std != 0:
                for j in range(len(data[:,i])):    
                    data[j,i] = (data[j,i] - mean) / std
        return data
    
    def getTrain(self, data):
        dataList = []
        for i in range(1):
            dataList.append(data.iloc[:, i+4:i+10])
        
        for i in dataList:
            i.columns = [0, 1, 2, 3, 4, 5]
       
        data = pd.concat(dataList, axis=0)
        
        return data

    def getValid(self, data):
        dataList = []
        dataList.append(data.iloc[:, 10:20])
        
        for i in dataList:
            i.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       
        data = pd.concat(dataList, axis=0)
        return data

    def getTest(self, data):
        #data = self.addone(data).values
        data = np.array(data).astype(float)
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        
        SQdata = data
        for i in np.nditer(SQdata):
            if i >= 0:
                i = math.sqrt(i)
        """
        TRIdata = data
        for i in np.nditer(TRIdata):
            if i >= 0:
                i = i**3
        """
        data = pd.DataFrame(data)
        SQdata = pd.DataFrame(SQdata)
        #TRIdata = pd.DataFrame(TRIdata)
        data = [data, SQdata]
        data = pd.concat(data, axis = 1)
        
        data = self.addone(data).values
        data = np.array(data).astype(float)
        
        return data
    
    def getTestHW1(self, data):
        #data = self.addone(data).values
        data = np.array(data).astype(float)
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        
        SQdata = data
        for i in np.nditer(SQdata):
            if i >= 0:
                i = math.sqrt(i)
        """
        TRIdata = data
        for i in np.nditer(TRIdata):
            if i >= 0:
                i = i**3
        """
        data = pd.DataFrame(data)
        SQdata = pd.DataFrame(SQdata)
        #TRIdata = pd.DataFrame(TRIdata)
        data = [data, SQdata]
        data = pd.concat(data, axis = 1)
        
        data = self.addone(data).values
        data = np.array(data).astype(float)
        
        return data

    def getTestBest(self, data):
        #data = self.addone(data).values
        data = np.array(data).astype(float)
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        
        SQdata = data
        for i in np.nditer(SQdata):
            if i >= 0:
                i = math.sqrt(i)
        """
        TRIdata = data
        for i in np.nditer(TRIdata):
            if i >= 0:
                i = i**3
        """
        data = pd.DataFrame(data)
        SQdata = pd.DataFrame(SQdata)
        #TRIdata = pd.DataFrame(TRIdata)
        data = [data, SQdata]
        data = pd.concat(data, axis = 1)
        
        #data = self.addone(data).values
        data = np.array(data).astype(float)
        
        return data

    def getTestOthers(self, data):
        concatList = []
        returnList = []
        counter = 0
        
        data = data.replace(['NR'],['0'])
        data = data.iloc[:, 4:]
        
        data1 = data
        for i in range(1, 4321):
            concatList.append(data1.iloc[i-1:i, :])
            if i % 18 == 0:
                for j in concatList:
                    j.index = [int(i/18)]
                returnList.append(pd.concat(concatList, axis = 1))
                concatList = []
        data = pd.concat(returnList, axis = 0)
        
        #data = self.addone(data).values
        #data = np.array(data).astype(float)
        """
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        
        SQdata = data
        for i in np.nditer(SQdata):
            if i >= 0:
                i = math.sqrt(i)
        
        TRIdata = data
        for i in np.nditer(TRIdata):
            if i >= 0:
                i = i**3
        
        data = pd.DataFrame(data)
        SQdata = pd.DataFrame(SQdata)
        #TRIdata = pd.DataFrame(TRIdata)
        data = [data, SQdata]
        data = pd.concat(data, axis = 1)
        """
        data = self.addone(data).values
        print(data)
        data = np.array(data).astype(float)
        
        return data
    
    def getTrainSet(self, data, k, index):
        splitList = np.array_split(data, k)        
        splitList = np.delete(splitList, index)
        
        for i in range(len(splitList) - 1):
            splitList[0] = np.concatenate((splitList[0],splitList[i+1]), axis = 0)
        
        return splitList[0]
        
    def getValidSet(self, data, k, index):
        splitList = np.array_split(data, k)
        return splitList[index]
        
    def addone(self, data):
        data.insert(0, 'megumin', 1)
        return data
    
