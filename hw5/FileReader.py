import csv
import numpy as np

class FileReader:
    def getUserID(self, filename):
        userID = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for ID in row:
                    userID.append(int(ID))
        return userID

    def getMovieID(self, filename):
        movieID = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for ID in row:
                    movieID.append(int(ID))
        return movieID

    def getMatrix(self, filename):
        matrix = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for rate in row:
                    matrix.append(int(rate))
        return matrix

    def readTrain(self, filename):
        data = []
        with open(filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                dataID, userID, movieID, rating = row
                data.append( [int(dataID), int(userID), int(movieID), int(rating)] )

        print('Train data len:', len(data))
        return np.array(data)

    def readTest(self, filename):
        data = []
        with open(filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                dataID, userID, movieID = row
                data.append( [int(dataID), int(userID), int(movieID)] )

        print('Train data len:', len(data))
        return np.array(data)

    
    def getIDandRating(self, data):
        print('Shuffle Data')
        np.random.seed(1019)
        index = np.random.permutation(len(data))
        data = data[index]
            
        userID = np.array(data[:, 1], dtype=int)
        movieID = np.array(data[:, 2], dtype=int)
        rating = np.array(data[:, 3], dtype=int)
        return userID, movieID, rating

    def getIDs(self, data):
        userID = np.array(data[:, 1], dtype=int)
        movieID = np.array(data[:, 2], dtype=int)
        return userID, movieID

    def getID(self, data):
        ID = np.array(data[:, 1:3], dtype=int)
        return ID

    def getUserAge(self, filename):
        data = []
        with open(filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                row[0] = row[0].split('::')
                data.append(int(row[0][2]))
        age = np.array(data, dtype = int)
        return age        
    
    def combineUserAndAge(self, user, age):
        new_user = []
        index = 0
        for i in range(len(user)):
            new_user.append([int(user[i]), int(age[index])])
            if i < 899872:
                if user[i] != user[i+1]:
                    index = index + 1
        return np.array(new_user)
                
