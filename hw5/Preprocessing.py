import sys
import csv

userID = []
movieID = []
matrix = []
def readAndPreprocess(filename):
    #store ID
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            #store all user ID
            userExist = False
            for ID in userID:
                if ID == row[1]:
                    userExist = True
                    break
            if not userExist:
                userID.append(row[1])
            #store all movie ID
            movieExist = False
            for ID in movieID:
                if ID == row[2]:
                    movieExist = True
                    break
            if not movieExist:
                movieID.append(row[2])
    csvfile.close()
    userID.pop(0)
    movieID.pop(0)
    
    #write ID to txt
    output = open('userID.txt', 'w')
    out = " ".join(userID)
    output.write(out + '\n')
    output = open('movieID.txt', 'w')
    out = " ".join(movieID)
    output.write(out + '\n')

def storeMatrix(user, movie, train):
    #load userID
    with open(user, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for i in reader:
            for j in i:
                userID.append(j)
    csvfile.close()
    
    #load movieID
    with open(movie, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for i in reader:
            for j in i:
                movieID.append(j)    
    csvfile.close()
    
    #store matrix
    for i in userID:
        print(i)
        vector = []
        for j in movieID:
            exist = False
            with open(train, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if row[1] == i and row[2] == j:
                        vector.append(row[3])
                        exist = True
                        break
                if not exist:
                    vector.append('-1')
            matrix.append(vector)
            csvfile.close()
    
    #write matrix to txt
    output = open('matrix.txt', 'w')
    for vector in matrix:
        out = " ".join(vector)
        output.write(out + '\n')

def storeMatrix2(user, movie, train):
    #load userID
    with open(user, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for i in reader:
            for j in i:
                userID.append(j)
    csvfile.close()
    
    #load movieID
    with open(movie, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for i in reader:
            for j in i:
                movieID.append(j)    
    csvfile.close()

    #store matrix
    with open(train, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        ID = userID[0]
        vector = ['-1'] * len(movieID)
        skip = True
        for row in reader:
            if not skip:
                if ID != row[1]:
                    ID = row[1]            
                    matrix.append(vector)
                    vector = ['-1'] * len(movieID)
                for i in range(len(movieID)):
                    if movieID[i] == row[2]:
                        vector[i] = row[3]
                        break
            skip = False
        matrix.append(vector)
    csvfile.close()

    #write matrix to txt
    output = open('matrix.txt', 'w')
    for vector in matrix:
        out = " ".join(vector)
        output.write(out + '\n')
  
storeMatrix2('userID.txt', 'movieID.txt', 'train.csv')

    
        


