import pandas as pd
import numpy as np

class dataProcessing:
    def getTrain(self, data):
        data = data.drop(data.columns[20:24], axis=1)
        data1 = data.iloc[:, :10]
        data2 = data.iloc[:, 10:]
        data2 = data2.drop(data2.index[96:240])
        data2.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        index = []
        for i in range(240,336):
            index.append(i)
        data2.index = index
        data = [data1, data2]
        data = pd.concat(data, axis=0)
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
        return data
    
    def getTest(self, data):
        data = data.drop(data.columns[:1], axis=1)
        return data
    
    def addone(self, data):
        data.insert(0, 'megumin', 1)
        return data
