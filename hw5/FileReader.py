import csv

class FileReader:
    def getUserID(self, filename):
        userID = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for ID in row:
                    userID.append([int(ID)])
        return userID

    def getMovieID(self, filename):
        movieID = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for ID in row:
                    movieID.append([int(ID)])
        return movieID

    def getMatrix(self, filename):
        matrix = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                for rate in row:
                    matrix.append(int(rate))
        return matrix
