import pandas as pd
import numpy as np
import math

class dataProcessing:        
    def getTrain(self, data):
        dataList = []
        for i in range(1):
            dataList.append(data.iloc[:, i:i+10])
        
        for i in dataList:
            i.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       
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
    
    def getDataSet(self, data):
        concatList = []
        returnList = []
        counter = 0
        
        data = data.iloc[1:, :9]
        data = data.replace(['NR'],['0'])
        data1 = data
        for i in range(1, 4321):
            concatList.append(data1.iloc[i-1:i, :])
            if i % 18 == 0:
                for j in concatList:
                    j.index = [int(i/18)]
                returnList.append(pd.concat(concatList, axis = 1))
                concatList = []
        data = pd.concat(returnList, axis = 0)
        """
        data = np.array(data).astype(float)
        
        for i in data:
            counter = counter + 1
            print(i[3])
            concatList = concatList + i
            if counter % 18 == 0:
                returnList.append(concatList)
                concatList = []
        
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
        data = np.array(data).astype(float)
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        
        return data

    def getTargetSet(self, data):
        PM25List = []
        counter = 0
        data = data.iloc[:, 9:].values
        for i in data:
            counter = counter + 1
            if counter % 18 == 10:
                PM25List.append(i[0])
        data = np.array(PM25List).astype(float)
        for i in np.nditer(data):
            if i < 0:
                i = 0.0
        return data
        
    def addone(self, data):
        data.insert(0, 'megumin', 1)
        return data
    
