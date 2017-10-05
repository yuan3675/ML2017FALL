import pandas as pd
import numpy as np

class dataProcessing:
    def getTrain(self, data):
        #data = data.drop(data.columns[20:24], axis=1)
        dataList = []
        for i in range(15):
            dataList.append(data.iloc[:, i:i+10])
        """
        data1 = data.iloc[:, :10]
        data2 = data.iloc[:, 1:11]
        data3 = data.iloc[:, 2:12]
        data4 = data.iloc[:, 3:13]
        data5 = data.iloc[:, 4:14]
        """
        #data2 = data2.drop(data2.index[96:240])
        for i in dataList:
            i.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        data2.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data3.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data4.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        data5.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        index = []
        for i in range(240,480):
            index.append(i)
        for i in dataList:
            i.index = index
        """
        data2.index = index
        data3.index = index
        data4.index = index
        data5.index = index
        """
        #data = [data1, data2, data3, data4, data5]
        data = pd.concat(dataList, axis=0)
        data[data < 0] = 0
        return data

    def getValid(self, data):
        data = data.drop(data.columns[20:24], axis=1)
        data = data.iloc[:, 10:]
        data = data.drop(data.index[:96])
        data.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = []
        for i in range(144):
            index.append(i)
        data.index = index
        data[data < 0] = 0
        return data
    
    def getTest(self, data):
        data = self.addone(data).values
        data = np.array(data).astype(float)
        return data
    
    def getDataSet(self, data):
        data = data.iloc[:, :9]
        #data[data < 0] = 0
        data = self.addone(data).values
        data = np.array(data).astype(float)
        return data

    def getTargetSet(self, data):
        data = data.iloc[:, 9:].values
        data = np.array(data).astype(float)
        return data
        
    def addone(self, data):
        data.insert(0, 'megumin', 1)
        return data
    
